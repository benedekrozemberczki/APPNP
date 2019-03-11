import torch
import random
import numpy as np
from tqdm import trange, tqdm
from torch_sparse import spmm
from texttable import Texttable
from appnp_layer import APPNPModel

class APPNPTrainer(object):


    def __init__(self, args, graph, features, target):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self.create_model()
        self.train_test_split()
        self.transfer_node_sets()
        self.process_features()
        self.transfer_features()

    def create_model(self):
        self.node_count = self.graph.number_of_nodes()
        self.number_of_labels = np.max(self.target)+1
        self.number_of_features = max([feature for node, features  in self.features.items() for feature in features]) + 1
        self.model = APPNPModel(self.args, self.number_of_labels, self.number_of_features, self.graph, self.device)
        self.model = self.model.to(self.device)

    def train_test_split(self):
        """
        Creating a train/test split.
        """
        random.seed(self.args.seed)
        nodes = [node for node in range(self.node_count)]
        random.shuffle(nodes)
        self.train_nodes = nodes[0:self.args.train_size]
        self.test_nodes = nodes[self.args.train_size:self.args.train_size+self.args.test_size]
        self.validation_nodes = nodes[self.args.train_size+self.args.test_size:]

    def transfer_node_sets(self):
        self.train_nodes = torch.LongTensor(self.train_nodes).to(self.device)
        self.test_nodes = torch.LongTensor(self.test_nodes).to(self.device)
        self.validation_nodes = torch.LongTensor(self.validation_nodes).to(self.device)

    def process_features(self):

        index_1 = [node for node in self.graph.nodes() for fet in self.features[node]]
        index_2 = [fet for node in self.graph.nodes() for fet in self.features[node]]
        values = [1.0/len(self.features[node]) for node in self.graph.nodes() for fet in self.features[node]]
        self.feature_indices = torch.LongTensor([index_1, index_2])
        self.feature_values = torch.FloatTensor(values)
        self.target = torch.LongTensor(self.target)


    def transfer_features(self):
        self.target = self.target.to(self.device)
        self.feature_indices = self.feature_indices.to(self.device)
        self.feature_values = self.feature_values.to(self.device)


    def score(self, index_set):
        self.model.eval()
        _, pred = self.model(self.feature_indices, self.feature_values).max(dim=1)
        correct = pred[index_set].eq(self.target[index_set]).sum().item()
        acc = correct / index_set.size()[0]
        return acc

    def do_a_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(self.feature_indices, self.feature_values)
        loss = torch.nn.functional.nll_loss(prediction[self.train_nodes], self.target[self.train_nodes])
        loss = loss + (self.args.lambd/2)*(torch.sum(self.model.layer_2.weight_matrix**2))
        loss.backward()
        self.optimizer.step()

    def train_neural_network(self):
        print("\nTraining.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.best_accuracy = 0
        self.step_counter = 0
        iterator = trange(self.args.epochs, desc='Validation accuracy: ', leave=True)
        for epoch in iterator:
            self.do_a_step()
            accuracy = self.score(self.validation_nodes)
            iterator.set_description("Validation accuracy: {:.4f}".format(accuracy))
            if accuracy >= self.best_accuracy:
                self.best_accuracy = accuracy
                self.test_accuracy = self.score(self.test_nodes)
                self.step_counter = 0
            else:
                self.step_counter = self.step_counter + 1
                if self.step_counter>self.args.early_stopping_rounds:
                    iterator.close()
                    break               

    def fit(self):
        self.train_neural_network()
        print("\nBreaking from training process because of early stopping.\n")
        print("Test accuracy: {:.4f}".format(self.test_accuracy))
        
