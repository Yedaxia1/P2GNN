import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.data import Batch, Data
from ProG.utils import act
import warnings
from deprecated.sphinx import deprecated


class BaseGNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT', num_class=2):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, hid_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, hid_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hid_dim) for i in range(gcn_layer_num-1)])

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool
        
        self.num_class = num_class
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(hid_dim, hid_dim // 2),     
        #     torch.nn.ReLU(True),
        #     torch.nn.Linear(hid_dim // 2, self.num_class)   
        # )

    def forward(self, graph_batch: Batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        for conv, bn in zip(self.conv_layers[0:-1], self.bns):
            x = conv(x, edge_index)
            # x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, 0.3, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        
        return graph_emb