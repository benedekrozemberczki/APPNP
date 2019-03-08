import torch
import random
from tqdm import trange, tqdm
from utils import create_propagator_matrix
from appnp_layer import APPNPModel
import numpy as np
from torch_sparse import spmm
from texttable import Texttable

class APPNPTrainer(object):


    def __init__(self, args, graph, features, target):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self.create_model()
        self.train_test_split()

    def create_model(self):
        self.node_count = self.graph.number_of_nodes()
        self.number_of_labels = np.max(self.target)+1
        self.number_of_features = max([feature for node, features  in self.features.items() for feature in features]) + 1
        self.model = APPNPModel(self.args, self.number_of_labels, self.number_of_features)
        self.model.to(self.device)

    def train_test_split(self):
        """
        Creating a train/test split.
        """
        random.seed(self.args.seed)
        nodes = [node for node in range(self.node_count)]
        random.shuffle(nodes)
        self.train_nodes = nodes[0:self.args.training_size]
        self.test_nodes = nodes[self.args.training_size:]


    def create_batches(self):
        random.shuffle(self.train_nodes)
        self.batches = [self.train_nodes[i:i+self.args.batch_size] for i in range(0, len(self.train_nodes), self.args.batch_size)]

    def process_batch(self, batch):
        features = np.zeros((len(batch), self.number_of_features))
        
        index_1 = [i for i, node in enumerate(batch) for feature_index in self.features[node]]
        index_2 = [feature_index for i, node in enumerate(batch) for feature_index in self.features[node]]
        value = [1.0 for i, node in enumerate(batch) for feature_index in self.features[node]]
        features[index_1,index_2] = value
        features = torch.FloatTensor(features).to(self.device)
        target = torch.LongTensor(np.array([self.target[node] for  node in batch])).to(self.device)
        return features, target


    def train_neural_network(self):
        print("\nTraining.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        for epoch in tqdm(range(self.args.epochs)):
            self.create_batches()
            for batch in self.batches:
                self.optimizer.zero_grad()
                features, target = self.process_batch(batch)
                prediction = self.model(features, self.args.dropout)
                loss = torch.nn.functional.nll_loss(prediction, target)
                loss = loss + (self.args.lambd/2)*(torch.sum(self.model.layer_1.weight_matrix**2))
                loss.backward()
                self.optimizer.step()

    def propagate(self):
        propagator = create_propagator_matrix(self.graph, self.args.alpha, self.args.model)
        if self.args.model=="exact":
            propagator = propagator.to(self.device)
            self.predictions = torch.mm(propagator, self.predictions)
        else:
            localized_predictions = self.predictions
            indices = propagator["indices"].to(self.device)
            weights = propagator["values"].to(self.device)
            for iteration in range(self.args.iterations):
                localized_predictions = (1-self.args.alpha)*spmm(indices, weights, localized_predictions.shape[0], localized_predictions)+self.args.alpha*self.predictions
            self.predictions = localized_predictions    


    def score(self):
        self.model.eval()
        print("\nScoring.\n")
        self.predictions = []
        for node in tqdm(self.graph.nodes()):
            features, target = self.process_batch([node])
            prediction = self.model(features, 0)
            self.predictions.append(prediction)
        self.predictions = torch.cat(self.predictions)

    def evaluate(self):
        self.predictions = torch.nn.functional.softmax(self.predictions,dim=1)
        values, indices = torch.max(self.predictions, 1)
        predictions = indices.cpu().detach().numpy()
        hits = predictions==self.target
 
        self.train_accuracy = round(sum(hits[self.train_nodes])/len(self.train_nodes),3)
        self.test_accuracy = round(sum(hits[self.test_nodes])/len(self.test_nodes),3)
        tab = Texttable() 
        tab.add_rows([["Train accuracy:",self.train_accuracy],["Test accuracy:",self.test_accuracy]])
        print(tab.draw())

    def fit(self):
        self.train_neural_network()
        self.score()
        self.propagate()
        self.evaluate()
