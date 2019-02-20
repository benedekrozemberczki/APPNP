import torch
import random
from tqdm import trange
from utils import create_propagator_matrix
from appnp_layer import PPNPLayer, APPNPLayer
from sklearn.model_selection import train_test_split

class PageRankNetwork(torch.nn.Module):
    """
    Page rank neural network class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    """
    def __init__(self, args, feature_number, class_number):
        super(PageRankNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.setup_layers()

    def setup_layer_structure(self):
        self.page_rank_convolution_1 = self.layer(self.feature_number, self.args.layers[0], self.args.iterations, self.args.alpha)
        self.page_rank_convolution_2 = self.layer(self.args.layers[0], self.args.layers[1], self.args.iterations, self.args.alpha)
        self.page_rank_convolution_3 = self.layer(self.args.layers[1], self.class_number, self.args.iterations, self.args.alpha)

    def setup_layers(self):
        if self.args.model == "exact":
            self.layer = PPNPLayer
        else:
            self.layer = APPNPLayer
        self.setup_layer_structure()

    def forward(self, propagation_matrix, input_features):

        propagation_matrix = torch.nn.functional.dropout(propagation_matrix, p = self.args.dropout, training = self.training)
        abstract_features_1 = self.page_rank_convolution_1(propagation_matrix, input_features, self.args.dropout, True)
        abstract_features_2 = self.page_rank_convolution_2(propagation_matrix, abstract_features_1, self.args.dropout, True)
        abstract_features_3 = self.page_rank_convolution_3(propagation_matrix, abstract_features_2, 0, False)
        predictions = torch.nn.functional.log_softmax(abstract_features_3, dim=1)

        return predictions

class APPNPTrainer(object):


    def __init__(self, args, graph, features, target):
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self.setup_features()
        self.setup_model()
        self.train_test_split()

    def train_test_split(self):
        nodes = [node for node in range(self.ncount)]
        random.shuffle(nodes)
        self.train_nodes = torch.LongTensor(nodes[0:self.args.training_size])
        self.validation_nodes = torch.LongTensor(nodes[self.args.training_size:self.args.training_size+self.args.validation_size])
        self.test_nodes = torch.LongTensor(nodes[self.args.training_size+self.args.validation_size:])

    def setup_features(self):

        self.ncount = self.graph.number_of_nodes()
        self.feature_number = self.features.shape[1]

        self.class_number = max(self.target)+1
        self.features = torch.FloatTensor(self.features.todense())
        self.target = torch.LongTensor(self.target)
        self.propagation_matrix = create_propagator_matrix(self.graph, self.args.alpha, self.args.model)

    def setup_model(self):
        self.model = PageRankNetwork(self.args, self.feature_number, self.class_number)

    def fit(self):
        accuracy = 0
        no_improvement = 0
        epochs = trange(self.args.epochs, desc="Accuracy")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        for epoch in epochs:
            self.optimizer.zero_grad()
            prediction = self.model(self.propagation_matrix, self.features)
            loss = torch.nn.functional.nll_loss(prediction[self.train_nodes], self.target[self.train_nodes])
            loss = loss + self.args.lambd*torch.sum(self.model.page_rank_convolution_1.weight_matrix**2)
            loss.backward()
            self.optimizer.step()
            new_accuracy = self.score(self.validation_nodes)
            epochs.set_description("Validation Accuracy=%g" % round(new_accuracy,4))
            if new_accuracy < accuracy:
                no_improvement = no_improvement + 1
                if no_improvement == self.args.early_stopping:
                    epochs.close()
                    break
            else:
                no_improvement = 0
                accuracy = new_accuracy               
        acc = self.score(self.test_nodes)
        print("\nTest accuracy: " + str(round(acc,4)) )

    def score(self, indices):
        self.model.eval()
        _, prediction = self.model(self.propagation_matrix, self.features).max(dim=1)
        correct = prediction[indices].eq(self.target[indices]).sum().item()
        acc = correct / indices.shape[0]
        return acc
