import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from Messagepass import  GCNConv
from torch_geometric.utils import to_undirected

class NGCF(nn.Module):
    def __init__(self,num_user,num_item,edge_index,reg_weight,dim_E):
        super(NGCF, self).__init__()
        self.num_user=num_user
        self.num_item=num_item
        self.edge_index=to_undirected(edge_index)
        self.reg_weight=reg_weight
        self.dim_E=dim_E

        self.layer1 = GCNConv(dim_E, dim_E)
        self.layer2 = GCNConv(dim_E, dim_E)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E), requires_grad=True))
        self.score = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E)))

    def forward(self):
        x1 = F.leaky_relu(self.layer1(self.id_embedding, self.edge_index))
        x2 = F.leaky_relu(self.layer2(x1, self.edge_index))
        self.score=torch.cat((x1,x2),dim=1)
    def loss(self,user,item):
        user=user.view(-1)
        item=item.view(-1)
        self.forward()
        user_score = self.score[user]
        item_score = self.score[item]
        score = torch.sum(user_score * item_score, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(score[0]-score[1])))
        reg_embedding_loss = (self.id_embedding[user] ** 2 + self.id_embedding[item] ** 2).mean()
        reg_loss = self.reg_weight * (reg_embedding_loss)

        return loss,reg_loss

    def full_accuracy(self, val_data,user_item, topk=10):
        user_tensor = self.score[:self.num_user]
        item_tensor = self.score[self.num_user:]

        all_index_of_rank_list= {}
        i=0
        for user in user_tensor:
            score_matrix=torch.matmul(user,item_tensor.t())
            _, index_of_rank_list = torch.topk(score_matrix, topk)
            index_of_rank_list=[ i+self.num_user for i in index_of_rank_list]
            all_index_of_rank_list[i]=index_of_rank_list
            i=i+1
        precision=recall=0.0
        length=0
        for u in val_data:
            pos_item=set(user_item[u])
            num_pos=len(pos_item)
            if num_pos==0:
                continue
            length+=1
            pred_item=set(all_index_of_rank_list[u])
            hit=len(pos_item.intersection(pred_item))

            precision+=float(hit/topk)
            recall+=float(hit/num_pos)

        return precision/length,recall/length












