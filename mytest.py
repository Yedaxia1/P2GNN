import os
import numpy as np
import pickle as pk
from torch_geometric.data import Data
from collections import defaultdict
import pickle as pk
from torch_geometric.utils import subgraph, k_hop_subgraph
import torch
import numpy as np
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon, Reddit
from torch_geometric.data import Data, Batch
import random
import warnings
from ProG.utils import mkdir
from random import shuffle
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T



dataname = 'ogbn-products'  # 'CiteSeer'  # 'PubMed' 'Cora' 'Computers'  'Reddit'   'ogbn-arxiv'  'ogbn-products'
# #
# pre_transform = SVDFeatureReduction(out_channels=100)
# if dataname in ['CiteSeer', 'PubMed', 'Cora']:
#     dataset = Planetoid(root='./dataset/', name=dataname)
#     data = dataset.data
# elif dataname=='Computers':
#     dataset = Amazon(root='./dataset/', name=dataname)
#     data = dataset.data
# elif dataname=='Reddit':
#     dataset = Reddit(root='./dataset/Reddit/', pre_transform=pre_transform)
#     data = dataset.data
# elif dataname in ['ogbn-arxiv', 'ogbn-products']:
#     dataset = PygNodePropPredDataset(name=dataname, root='./dataset/', pre_transform=pre_transform)
#     dataname = '_'.join(dataname.split('-')) 
#     data = dataset.data
#     data.y = data.y.view(-1)
    
    
data = pk.load(open('./dataset/{}/feature_reduced.data'.format('ogbn_products'), 'br'))
from collections import Counter
y = data.y.numpy().tolist()
print(y)
total = len(y)
result = Counter(y)
print(result)
for cid, n in result.items():
    print("class-{}: {}%".format(cid, round(100*n/total, 2)))
# data_path = os.path.join('dataset/ogbn_arxiv/index/task0.meta.test.query')
    
# with open(data_path, 'br') as t1s:
#     t1s = pk.load(t1s)
    
#     print(t1s)
#     t1s = t1s['pos']
#     print(t1s)
#     print(len(t1s))
    
#     # for data in t1s:
#     #     x = data.x.detach()
#     #     edge_index = data.edge_index
#     #     data_ = Data(x=x, edge_index=edge_index)
        
#     #     print(data_)

# data = pk.load(open('./dataset/{}/feature_reduced.data'.format('ogbn_arxiv'), 'br'))
# print(data.y.view(-1))


