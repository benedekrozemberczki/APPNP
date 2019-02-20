import math
import torch
from torch_sparse import spmm

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class AbstractPPNPLayer(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param :
    :param :
    :param :
    :param :
    """
    def __init__(self, in_channels, out_channels, iterations, alpha):
        super(AbstractPPNPLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.alpha = alpha
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels,self.bias)

class PPNPLayer(AbstractPPNPLayer):
    """
    Exact personalized PageRank convolution layer.
    """
    def forward(self, personalized_page_rank_matrix, features, dropout_rate, transform, density):
        if density:
            filtered_features = torch.mm(features, self.weight_matrix)
        else:
            filtered_features = spmm(features["indices"], features["values"], features["dimensions"][0],  self.weight_matrix)
        filtered_features = torch.nn.functional.dropout(filtered_features , p = dropout_rate, training = self.training)
        if transform:
            filtered_features = torch.nn.functional.relu(filtered_features)
        localized_features = torch.mm(personalized_page_rank_matrix, filtered_features)
        localized_features = localized_features + self.bias
        return localized_features

class APPNPLayer(AbstractPPNPLayer):
    """
    Approximate personalized PageRank Convolution Layer.
    """
    def forward(self, normalized_adjacency_matrix, features, dropout_rate, transform, density):
        if density:
            base_features = torch.mm(features, self.weight_matrix)
        else:
            base_features = spmm(features["indices"], features["values"], features["dimensions"][0],  self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features, p = dropout_rate, training = self.training)
        if transform:
            base_features = torch.nn.functional.relu(base_features) + self.bias
        localized_features = base_features
        for iteration in range(self.iterations):
            localized_features = (1-self.alpha)*spmm(normalized_adjacency_matrix["indices"], normalized_adjacency_matrix["values"], localized_features.shape[0], localized_features)+self.alpha*base_features


        return localized_features
