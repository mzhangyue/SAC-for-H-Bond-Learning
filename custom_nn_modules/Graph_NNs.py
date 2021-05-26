#import math
import torch
import torch.nn as nn
#import torch.nn.functional as F
#from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

####################
## Helper modules ##
####################

# Multilayer perceptron
class MLP(Module):
    def __init__(self, in_dim, linear_units, activation=None, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for d0, d1 in zip([in_dim]+linear_units[:-1], linear_units):
            layers.append(nn.Linear(d0, d1))
            layers.append(nn.Dropout(dropout))
        if activation is not None:
            layers.append(activation)
        self.net = nn.Sequential(*layers)
        
    def forward(self, input):
        return self.net(input)
    
# One graph convolutional layer
# in_features (int): num_input_features
# hidden_features (int): num_hidden_features
# edge_dim (int): dimension of edge
# activation (function): activation function
# dropout (float): probability of drop out 
class GraphConvolutionLayer(Module):
    def __init__(self, in_features, hidden_features, edge_dim, activation, dropout):
        super(GraphConvolutionLayer, self).__init__()
        self.edge_dim = edge_dim
        self.hidden_features = hidden_features
        self.adj_list = nn.ModuleList()
        for _ in range(self.edge_dim):
            self.adj_list.append(nn.Linear(in_features, hidden_features))
        self.residual= nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, adj, hidden=None):
        if hidden is not None:
            input1 = torch.cat((input, hidden), -1)
        else:
            input1 = input
        # For each edge_dim type output a linear combination of hidden
        hidden1 = torch.stack([self.adj_list[i](input1) for i in range(self.edge_dim)], 1)
        # propagate info via adjacent vertices
        hidden1 = torch.matmul(adj, hidden1)
        # Combine info and output residual
        out = torch.sum(hidden1, 1) + self.residual(input1)
        # non-linear activation and dropout
        out = self.activation(out) if self.activation is not None else out
        out = self.dropout(out)
        return out



# One graph convolutional module consisting of many graph convolutional layers
# in_features (int):
# units (array of int): the ouput dim of each GCN layer
# edge_dim (float): edge dimension
# dropout (float): probability of dropout
class GraphConvolution(Module):
    def __init__(self, in_features, units, edge_dim, dropout=0):
        super(GraphConvolution, self).__init__()
        activation = nn.Tanh()
        self.units = units
        self.convs = nn.ModuleList()
        layer_dims = list([d + in_features for d in units])
        # This code shifts layer_dims to pair input and output dims of each layer
        for d0, d1 in zip([in_features]+layer_dims[:-1], units): 
            self.convs.append(GraphConvolutionLayer(d0, d1, edge_dim, activation, dropout))
    
    # Apply each GCN to the input graph and store the result
    def forward(self, input, adj, hidden=None):
        hidden1 = hidden
        for idx in range(len(self.units)):
            hidden1 = self.convs[idx](input, adj, hidden1)
        return hidden1

# Aggregates the outputs of all the GCNs
# in_features (int):
# out_features (int):
# node_dim (int):
# dropout (float):
class GraphAggregation(Module):
    def __init__(self, in_features, out_features, node_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+node_dim, out_features), nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+node_dim, out_features), nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        # \sum Sigmoid(Linear(input)) x Tanh(Linear(input)) 
        output = torch.sum(torch.mul(i, j), 1)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)
        
        return output