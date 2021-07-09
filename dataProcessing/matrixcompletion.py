import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import to_undirected, add_self_loops, subgraph, is_undirected
from os import listdir, mkdir
import os.path as osp
import json
import numpy as np
#from utils import get_neighbors
import json
from random import shuffle
from random import sample
import random
import gzip
import re


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2
                 for (idx, char) in enumerate(ALPHABET)
                 }  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1


class matrixcompletionGraphData(Data):
    def __init__(self):
        super(matrixcompletionGraphData, self).__init__()

    """
    def __init__(self, edge_index, edge_index_three_hop, edge_index_five_hop, label, x):
        super(GraphData, self).__init__()
        self.edge_index = edge_index
        self.edge_index_three_hop = edge_index_three_hop
        self.edge_index_five_hop = edge_index_five_hop
        self.label = label
        self.x = x
    """

    def load_attr(self, edge_index, x, slot_id, label, x_validate, label_validate, slot_id_validate):
        """由于dataset的限制，不能初始化的时候传入参数，所以设立一个函数传入参数。

        Args:
            edge_index ([type]): ast one hop edge index
            edge_index_three_hop ([type]): ast three hop edge index
            edge_index_five_hop ([type]): ast five hop edge index
            value_label ([type]): the label of node value
            type_label ([type]): the label of node type
            x ([type]): the initial embedding of node value
            x_type ([type]): the initial embedding of node type
            right_most ([type]): the right most node id of a graph
        """
        # 一行 节点值为0， 然后目标节点值为分数。有边连接为1.没有分数为0，待预测节点为0.与所有有关的节点连接起来。
        self.edge_index = edge_index
        self.x = x
        self.slot_id = slot_id
        self.label=label
        self.x_validate=x_validate
        self.label_validate=label_validate
        self.slot_id_validate=slot_id_validate
        #初始特征维度为1.

    def __inc__(self, key, value):
        if key == "slot_id":
            return self.num_nodes
        return super().__inc__(key, value)

    def __cat_dim__(self, key, value):
        if key == "label":
            return 0
        return super().__cat_dim__(key, value)

def matrixcompletionMake_task_input(batch_data, get_num_edge_types=False):
    # 当get_num_edge_types设置为true的时候返回边的个数。
    if get_num_edge_types:
        return 1
    return {
        "x": batch_data.x,
        "edge_list":[
            batch_data.edge_index,
            
            #batch_data.LastLexicalUse_index,
            #batch_data.ComputedFrom_index,
            #batch_data.GuardedByNegation_index,
            #batch_data.GuardedBy_index,
            #batch_data.FormalArgName_index,
            #batch_data.ReturnsTo_index,
            #batch_data.SelfLoop_index,
        ],
        "slot_id":batch_data.slot_id,
    }

class matrixcompeltionGraphDataset(Dataset):
    """静态的learning graph dataset。做出以下改进：
        1.将图的大小限制到0-150
        2.每个图都有self loop。因为是针对于attention的。所以加上self loop后，更容易计算邻居对于本节点的权重。
            当没有邻居节点时，该节点值也会自我更新。不会因为没有邻居节点，节点值就为空。

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
                 root,
                 rows=20,
                 columns=400,
                 data_per_column_rate=0.2,
                 target_per_data_rate=0.5,
                 transform=None,
                 pre_transform=None):
        # 200*4000
        # 3000*3000
        self.rows=rows
        self.columns=columns
        self.data_per_column_rate=data_per_column_rate
        self.target_per_data_rate=target_per_data_rate
        super(matrixcompeltionGraphDataset, self).__init__(root, transform, pre_transform)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        if not osp.exists(self.processed_dir):
            mkdir(self.processed_dir)
        return [
            file for file in listdir(self.processed_dir)
            if file.startswith("data")
        ]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                                   'data_{}.pt'.format(idx)))
        return data

    '''
    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        ...'''

    def trans_list_to_edge_tensor(self,
                                  edge_index,
                                  is_add_self_loop=True,
                                  is_to_undirected=True, self_loop_num_nodes:int=None, make_sub_graph=True, do_node_trans=False):
        """将[src,tgt]转为输入到data对象中的edge_index，即tensor([src...],[tgt...])
            加上自连接边、加上双向边。

        Args:
            edge_index ([type]): [src,tgt]
            add_self_loop (bool, optional): 是否加自连接边. Defaults to True.
            to_undirected (bool, optional): 是否转为无向图. Defaults to True.
        """
        if edge_index is not None and len(edge_index) != 0:  # 当edge_index存在且有边时。   
            t = torch.tensor(edge_index).t().contiguous()
        else:
            t = torch.tensor([]).type(torch.LongTensor).t().contiguous()
        

        if is_add_self_loop:
            if self_loop_num_nodes:
                t, weight = add_self_loops(t, num_nodes=self_loop_num_nodes)
            else:
                t, weight = add_self_loops(t, num_nodes=self.columns)  # 这里强制加了0-150的节点。方便做subgraph不会报错。
        if is_to_undirected and t.shape[0]!=0:
            t = to_undirected(t)
        
        if make_sub_graph:
            t, attr = subgraph(torch.arange(start=0, end=self.columns), t)

        if do_node_trans:
            t_new = torch.tensor([[self.node_trans_dict[i.item()] for i in row]  for row in t]).type(torch.LongTensor)
            t = t_new
        return t

    def process(self):
        # Read data into huge `Data` list.
        data_i = 0

        for data_i in range(self.rows):
            # 创造max_graph个数据。
            # 按照这样的规则创造：
            #   1.每个数据一行。M列。其中N个有数据。数据范围为[0,1].
            #   2.每个有数据的点和该行构成边。
            #   3.选择一定百分比（初始50%）有数据的点设置为1，目标就是为了预测这一定的数据中的值。
            #   4.有一个评分计算函数。然后和真正的评分进行计算。
            #   5.损失函数用MSELoss来计算。节点的distribution获取为随机生成的值。
            
            # 生成一行数据。
            # 1. 为全随机
            # 2. 然后选择部分值为0
            # 气死。更本不想写。
            # 步骤：
            # 1. 生成columns行数据，设置为0.
            # 2. 随机选择N个数据设置为随机数。
            # 3. 选择N个数据中的一定个，将他们的值设置为target。并将他们值设置为1.
            # 4，对所有N个数据建立边。

            x = torch.zeros(size=(self.columns, self.columns)).type(torch.FloatTensor)
            x_validate = torch.zeros(size=(self.columns, self.columns)).type(torch.FloatTensor)

            edge_index = []
            #edge_index.append([0,0])
            label = []
            label_validate = []
            slot_id = []  # slot_id 就是target节点

            N = int(self.data_per_column_rate*self.columns)
            target = int(self.target_per_data_rate*N)

            N_index = np.random.choice(self.columns, N, replace=False)
            slot_id = N_index[:target]
            slot_id_validate=N_index[target:]

            for index in N_index[:target]:
                x[0][index] = 1
                edge_index.append([0, index])
                #edge_index.append([index, 0])
                #edge_index.append([index, index])
                label.append(np.random.random(1)[0])


            for index in N_index[target:]:
                x_validate[0][index] = np.random.random(1)[0]
                edge_index.append([0, index])
                #edge_index.append([index, 0])
                #edge_index.append([index, index])

            ## validate
            for index in N_index[:target]:  
                x_validate[0][index] = np.random.random(1)[0]
                #edge_index.append([0, index])
                #edge_index.append([index, 0])
                #edge_index.append([index, index])


            for index in N_index[target:]:
                x[0][index] = 1
                #edge_index.append([0, index])
                #edge_index.append([index, 0])
                #edge_index.append([index, index])
                label_validate.append(np.random.random(1)[0])

            
            edge_index_tensor = self.trans_list_to_edge_tensor(edge_index)
            label = torch.FloatTensor([label])
            label_validate = torch.FloatTensor([label_validate])
            slot_id = torch.LongTensor([slot_id])
            slot_id_validate = torch.LongTensor([slot_id_validate])

            data = matrixcompletionGraphData()
            data.load_attr(edge_index_tensor, x, slot_id, label, x_validate, label_validate, slot_id_validate)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(data_i)))



            





if __name__ == '__main__':
    from torch_geometric.data import DataLoader


    gd = matrixcompeltionGraphDataset("data/random/")
    for g in gd:
        print(g)
    
