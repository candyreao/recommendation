import csv
import os
import torch
'''a=[]
f = open(r"item.csv","r",encoding="utf-8")
reader = csv.reader(f)
for row in reader:
    a.append(row[0])

print(len(a))
f1 = open(r"ratings_matrix.csv","r",encoding="utf-8")
reader1 = csv.reader(f1)
for row in reader1:
    b=row
    break
print(len(b))
c=[]
for i in b:
    if i in a:
        c.append(i)
print(len(c),c)


feature=pd.read_csv('C:\\Users\\ilearn\Desktop\\recommendation\\movielens\\posterfeature.csv', index_col=0)
print(feature)
feature['f']=feature['f'].apply(literal_eval)
print(feature)
#print(feature[1:])
#print(feature['10'])'''
a=torch.tensor([[1.0,2.0],[1.0,3.0]])
b=torch.tensor([[1.0,2.0], [-1.0,4.0]])
c=a*b
print(c)
score = torch.sum(c, dim=1).view(-1,2)
print(score)
print(score[0],score[1])
