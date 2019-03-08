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

class FullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(FullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        """
        Doing a forward pass.
        :param features: Feature matrix.
        :return filtered_features: Convolved features.
        """
        filtered_features = torch.mm(features, self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class APPNPModel(torch.nn.Module):

    def __init__(self, args, number_of_labels, number_of_features):
        super(APPNPModel, self).__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.setup_layers()

    def setup_layers(self):
        self.layer_1 = FullyConnected(self.number_of_features, self.args.layers[0])
        self.layer_2 = FullyConnected(self.args.layers[1], self.number_of_labels)

    def forward(self, features, dropout_rate):
        features = torch.nn.functional.dropout(features, p = dropout_rate, training = self.training)
        latent_features_1 = torch.nn.functional.relu(self.layer_1(features))
        latent_features_1 = torch.nn.functional.dropout(latent_features_1, p = dropout_rate, training = self.training)
        latent_features_2 = torch.nn.functional.softmax(self.layer_2(latent_features_1),dim=1)
        return latent_features_2

