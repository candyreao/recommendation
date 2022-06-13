import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import random
import csv
import torch.utils.data as data
import torch.optim as optim
from dataset import Traindataset
from GRCN import GRCN


def for_edge_index(data,user_list,item_list,user_num):
    flag = 0
    edge_index=[]
    user_item={}
    data_index = data.index
    for x in data_index:
        if flag == 0:
            flag = 1
            continue
        user = data.loc[x, :]
        x = user_list.index(x)
        i = list(user[user.values > 0].index)
        pos_item=[]
        for ii in i:
            egde=[]
            if ii in item_list:
                ii=item_list.index(ii)+user_num
                egde.append(x)
                egde.append(ii)
                edge_index.append(egde)
                pos_item.append(ii)
            else:
                continue
        user_item[x]=pos_item
    edge_index=torch.tensor(edge_index)
    return edge_index,user_item

if __name__ == '__main__':
    train = pd.read_csv('/home/share/wangjingwen/recommendation/movielens/test.csv', index_col=0)
    v_f = torch.load('poster.pt', map_location='cpu')
    v_f= torch.tensor(v_f, dtype=torch.float)
    t_f= torch.load('comment.pt', map_location='cpu')
    t_f= torch.tensor(t_f, dtype=torch.float)
    item_list =[]
    f = open(r"item.csv", "r", encoding="utf-8")
    reader = csv.reader(f)
    for row in reader:
        item_list.append(row[0])
    user_list = train.index.drop('userId').tolist()
    user_num = 610
    item_num = len(item_list)
    edge_index,user_item = for_edge_index(train, user_list, item_list,user_num)
    train_dataset = Traindataset(user_num,item_num,edge_index,user_item)
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = GRCN(user_num,item_num,edge_index.T,0.001,v_f,t_f)
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': 0.001}])

    for epoch in range(100):
        model.train()
        sum_loss = 0.0
        sum_loss_reg=0.0
        step=0
        for u,ij in train_loader:
            optimizer.zero_grad()
            loss, loss_reg = model.loss(u,ij)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_loss_reg+=loss_reg.item()
            step+=1
        print(sum_loss/step,sum_loss_reg/step)

    users = set(range(0, user_num))
    val=random.sample(users, 50)
    precision, recall=model.full_accuracy(val,user_item,10)
    print(precision, recall)



