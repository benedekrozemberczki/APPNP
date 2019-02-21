import torch
import random
from tqdm import trange
from utils import create_propagator_matrix
from appnp_layer import PPNPLayer, APPNPLayer

class PageRankNetwork(torch.nn.Module):
    """
    Page rank neural network class.
    :param args: Arguments object.
    :param feature_number: Number of features.
    :param class_number: Number of target classes.
    """
    def __init__(self, args, feature_number, class_number):
        super(PageRankNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.setup_layers()

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional layers).
        """
        self.page_rank_convolution_1 = self.layer(self.feature_number, self.args.layers[0], self.args.iterations, self.args.alpha)
        self.page_rank_convolution_2 = self.layer(self.args.layers[0], self.args.layers[1], self.args.iterations, self.args.alpha)
        self.page_rank_convolution_3 = self.layer(self.args.layers[1], self.class_number, self.args.iterations, self.args.alpha)

    def setup_layers(self):
        """
        Deciding the type of layer used.
        """
        if self.args.model == "exact":
            self.layer = PPNPLayer
        else:
            self.layer = APPNPLayer
        self.setup_layer_structure()

    def forward(self, propagation_matrix, input_features):
        """
        Making a forward pass for propagation.
        :param propagation_matrix: Propagation matrix (normalized adjacency or personalized pagerank).
        :param input_features: Input node features.
        :return predictions: Prediction vector.
        """
        if self.args.model == "exact":
            propagation_matrix = torch.nn.functional.dropout(propagation_matrix, p = self.args.dropout, training = self.training)
        abstract_features_1 = self.page_rank_convolution_1(propagation_matrix, input_features, self.args.dropout, True, False)
        abstract_features_2 = self.page_rank_convolution_2(propagation_matrix, abstract_features_1, self.args.dropout, True, True)
        abstract_features_3 = self.page_rank_convolution_3(propagation_matrix, abstract_features_2, 0, False, True)
        predictions = torch.nn.functional.log_softmax(abstract_features_3, dim=1)
        return predictions

class APPNPTrainer(object):
    """
    Class for training the neural network.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :param features: Feature sparse matrix.
    :param target: Target vector.
    """
    def __init__(self, args, graph, features, target):
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self.setup_features()
        self.setup_model()
        self.train_test_split()

    def train_test_split(self):
        """
        Creating a train/test split.
        """
        random.seed(self.args.seed)
        nodes = [node for node in range(self.ncount)]
        random.shuffle(nodes)
        self.train_nodes = torch.LongTensor(nodes[0:self.args.training_size])
        self.validation_nodes = torch.LongTensor(nodes[self.args.training_size:self.args.training_size+self.args.validation_size])
        self.test_nodes = torch.LongTensor(nodes[self.args.training_size+self.args.validation_size:])

    def setup_features(self):
        """
        Creating a feature matrix, target vector and propagation matrix.
        """"
        self.ncount = self.features["dimensions"][0]
        self.feature_number = self.features["dimensions"][1]
        self.class_number = max(self.target)+1
        self.target = torch.LongTensor(self.target)
        self.propagation_matrix = create_propagator_matrix(self.graph, self.args.alpha, self.args.model)

    def setup_model(self):
        """
        Defining a PageRankNetwork.
        """
        self.model = PageRankNetwork(self.args, self.feature_number, self.class_number)

    def fit(self):
        """
        Fitting a neural network with early stopping.
        """
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
            epochs.set_description("Validation Accuracy: %g" % round(new_accuracy,4))
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
        """
        Scoring a neural network.
        :param indices: Indices of nodes involved in accuracy calculation.
        :return acc: Accuracy score.
        """
        self.model.eval()
        _, prediction = self.model(self.propagation_matrix, self.features).max(dim=1)
        correct = prediction[indices].eq(self.target[indices]).sum().item()
        acc = correct / indices.shape[0]
        return acc
