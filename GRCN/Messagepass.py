import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import uniform
import math
import numpy
class GCNConv1(MessagePassing):
    def __init__(self,in_channels,out_channels,flow="target_to_source",normalize=True,aggr='add'):
        super(GCNConv1, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.aggr=aggr

    def forward(self,x,edge_index):
        return self.propagate(edge_index,size=(x.size(0), x.size(0)), x=x)
    def message(self, edge_index_i, x_i, x_j, size_i):
        self.p=torch.mul( x_i,x_j).sum(dim=-1)
        self.p=softmax(self.p,edge_index_i,size_i)
        print(self.p)
        return x_j*self.p.view(-1,1)
    def update(self, x,aggr_out):
        return x+aggr_out


class GCNConv2(MessagePassing):
    def __init__(self,in_channels,out_channels,flow="target_to_source",normalize=True,aggr='add'):
        super(GCNConv2, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.aggr=aggr

    def forward(self,x,edge_index,s):
        self.s=s
        return self.propagate(edge_index,size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j*self.s
    def update(self, x,aggr_out):
        return aggr_out