import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn.inits import uniform
class GCNConv(MessagePassing):
    def __init__(self,in_channels,out_channels,flow="target_to_source",normalize=True,aggr='add'):
        super(GCNConv, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.weight1=Parameter(torch.Tensor(self.in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(self.in_channels, out_channels))
        uniform(self.in_channels, self.weight1)
        uniform(self.in_channels, self.weight2)

    def forward(self,x,edge_index):

        row, col = edge_index
        deg=degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm.view(-1,1)
        return self.propagate(edge_index,size=(x.size(0), x.size(0)), x=x,norm=norm)
    def message(self, x_i,x_j,norm):
        x_j=torch.matmul(x_j, self.weight1) + torch.matmul(x_i.mul(x_j), self.weight2)
        x_j=norm*x_j
        return x_j
    def update(self, x,aggr_out):
        x=torch.matmul(x, self.weight1)
        return x+aggr_out