import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from Messagepass import  GCNConv1,GCNConv2
from torch_geometric.utils import to_undirected

class GCN1(nn.Module):
    def __init__(self,num_user,num_item,edge_index,dim_E,dim_latent):
        super(GCN1, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.dim_E=dim_E
        self.dim_latent=dim_latent

        self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True))
        self.MLP = nn.Linear(self.dim_E, self.dim_latent,bias=True)

        self.conv_embed_1 = GCNConv1(self.dim_latent, self.dim_latent)
        self.conv_embed_2 = GCNConv1(self.dim_latent, self.dim_latent)


    def forward(self,features):
        new_features = self.MLP(features)

        x = torch.cat((self.preference, new_features), dim=0)
        x = F.normalize(x)
        x1 = self.conv_embed_1(x, self.edge_index)
        x2= self.conv_embed_2(x1, self.edge_index)

        return x2,self.conv_embed_2.p


class GCN2(nn.Module):
    def __init__(self,num_user,num_item,edge_index,dim_latent):
        super(GCN2, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.dim_latent=dim_latent
        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, self.dim_latent), requires_grad=True))
        self.conv_embed_1=GCNConv2(self.dim_latent, self.dim_latent)
        self.conv_embed_2 = GCNConv2(self.dim_latent, self.dim_latent)

    def forward(self,s):

        x=self.conv_embed_1(self.id_embedding,self.edge_index,s)
        x1=self.conv_embed_2(x,self.edge_index,s)
        return x+x1


class GRCN(nn.Module):
    def __init__(self,num_user,num_item,edge_index,reg_weight,v_f,t_f):
        super(GRCN, self).__init__()
        self.num_user=num_user
        self.num_item=num_item
        self.edge_index=to_undirected(edge_index)
        self.reg_weight=reg_weight
        self.v_f = torch.tensor(v_f, dtype=torch.float)
        self.v_gcn = GCN1(num_user,num_item,edge_index,self.v_f.size(1),256)
        self.t_f = torch.tensor(t_f, dtype=torch.float)
        self.t_gcn = GCN1(num_user,num_item,edge_index,self.t_f.size(1),256)
        self.rou = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, 2))))

        self.e_gcn=GCN2(num_user,num_item,edge_index,128)

        self.score = nn.init.xavier_normal_(torch.rand((num_user + num_item, 128)))

    def forward(self):
        v,v_s=self.v_gcn(self.v_f)
        t,t_s = self.t_gcn(self.t_f)
        f_embed=torch.cat((v,t),dim=1)
        s_weight=torch.cat(v_s,t_s)
        rou=torch.cat((self.rou[self.edge_index[0]],self.rou[self.edge_index[1]]),dim=0)
        s, _ = torch.max(s_weight*rou, dim=1)
        s=s.view()

        id_embed=self.e_gcn(s)

        self.score=torch.cat((id_embed,f_embed),dim=1)
    def loss(self,user,item):
        user=user.view(-1)
        item=item.view(-1)
        self.forward()
        user_score = self.score[user]
        item_score = self.score[item]
        score = torch.sum(user_score * item_score, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(score[0]-score[1])))
        reg_embedding_loss = (self.e_gcn.id_embedding[user] ** 2 + self.e_gcn.id_embedding[item] ** 2).mean()
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












