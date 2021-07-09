from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
from layers import EmbeddingLayer, MPLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple
#from ..fed_client import TaskFold
from utils import TaskFold


class GGNN(MessagePassing):
    def __init__(
        self,
        num_edge_types,
        in_features,
        out_features,
        dropout=0,
        add_self_loops=False,
        bias=True,
        aggr="mean",
        device="cpu",
    ):
        super(GGNN, self).__init__(aggr=aggr)
        # params set
        self.num_edge_types = num_edge_types
        self.device = device
        #self.output_model = output_model.lower()
        self.dropout = dropout
        # 先对值进行embedding
        self.value_embeddingLayer = nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU())

        self.MessagePassingNN = nn.ModuleList([
            MPLayer(in_features=out_features, out_features=out_features, device=device)
            for _ in range(self.num_edge_types)
        ])

        self.gru_cell = torch.nn.GRUCell(input_size=out_features, hidden_size=out_features)
        self.output_layer = nn.Linear(out_features, 1)

    def forward(self,
                x,
                edge_list: List[torch.tensor],
                slot_id):

        x_embedding = self.value_embeddingLayer(x)

        last_node_states = x_embedding
        for _ in range(8):
            out_list = []
            cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
            for i in range(len(edge_list)):
                edge = edge_list[i]
                if edge.shape[0] != 0:
                    # 该种类型的边存在边
                    out_list.append(F.relu(self.MessagePassingNN[i](cur_node_states, edge)))
            cur_node_states = sum(out_list)
            new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
            last_node_states = new_node_states

        out = last_node_states
        
        out_score = out[slot_id]
        out_score = self.output_layer(out_score)
        out_score = torch.squeeze(out_score, dim=-1)
        return out_score