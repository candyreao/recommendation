import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
class Traindataset(Dataset):
    def __init__(self, num_user, num_item,edge_index,user_item):
        self.edge_index = edge_index
        self.user_item=user_item
        self.num_user = num_user
        self.num_item = num_item
        self.item = set(range(num_user, num_user + num_item))

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item= self.edge_index[index]
        user=user.item()
        pos_item=pos_item.item()
        while True:
            neg_item = random.sample(self.item, 1)[0]

            if neg_item not in self.user_item[user]:
                break
        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item])


