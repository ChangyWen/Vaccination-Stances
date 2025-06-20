#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dgl
from dgl.data import DGLDataset
import torch
from twitter_func import create_nx_graph
import pickle
import random

nx_graph_file = '../data/nx_graph1031_wcc1.pkl'
to_homo_file = '../data/dgl1031_homo_wcc1.pkl'
to_hetero_file = '../data/dgl1031_hetero_wcc1.pkl'
# nx_graph_file = '../data/nx_graph1231_wcc1.pkl'
# to_homo_file = '../data/dgl1231_homo_wcc1.pkl'
# to_hetero_file = '../data/dgl1231_hetero_wcc1.pkl'

class Twitter_homo(DGLDataset):
    def __init__(self, ):
        super().__init__(name='Twitter_homo')

    def process(self):
        random.seed(0)

        '''load networkx graph'''
        with open(nx_graph_file, "rb") as f:
            nx_graph = pickle.load(f)

        self.graph = dgl.from_networkx(nx_graph, node_attrs=['feat', 'labels'], edge_attrs=['feat'])

        n_nodes = len(nx_graph.nodes)
        n_train = int(n_nodes * 0.8)
        n_val = int(n_nodes * 0.1)
        tmp = [0] * n_train + [1] * n_val + [2] * (n_nodes - n_train - n_val)
        random.shuffle(tmp)
        tmp = torch.tensor(tmp)
        train_mask = (tmp == 0)
        val_mask = (tmp == 1)
        test_mask = (tmp == 2)
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class Twitter_hetero(DGLDataset):
    def __init__(self, ):
        super().__init__(name='Twitter-Covid-19-Vaccination-Stance')

    def process(self):
        random.seed(0)

        '''load networkx graph'''
        with open(nx_graph_file, "rb") as f:
            nx_graph = pickle.load(f)

        graph_tmp = dgl.from_networkx(nx_graph, node_attrs=['feat', 'labels'], edge_attrs=['feat'])

        nodes = list(nx_graph.nodes)
        edges = list(nx_graph.edges)

        from_nodes, to_nodes = [], []
        for e in edges:
            from_nodes.append(e[0])
            to_nodes.append(e[1])

        data_dict = {
            ('user', 'to', 'user'): (from_nodes, to_nodes),
            ('user', '_to', 'user'): (to_nodes, from_nodes),
        }
        num_nodes_dict = {'user': len(nodes)}

        self.graph = dgl.heterograph(data_dict, num_nodes_dict)

        self.graph.edges['to'].data['feat'] = graph_tmp.edata['feat']
        self.graph.edges['_to'].data['feat'] = graph_tmp.edata['feat']
        self.graph.nodes['user'].data['feat'] = graph_tmp.ndata['feat']
        self.graph.nodes['user'].data['labels'] = graph_tmp.ndata['labels']

        n_nodes = len(nx_graph.nodes)
        n_train = int(n_nodes * 0.8)
        n_val = int(n_nodes * 0.1)
        tmp = [0] * n_train + [1] * n_val + [2] * (n_nodes - n_train - n_val)
        random.shuffle(tmp)
        tmp = torch.tensor(tmp)
        train_mask = (tmp == 0)
        val_mask = (tmp == 1)
        test_mask = (tmp == 2)
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


if __name__ == '__main__':
    # create_nx_graph()

    g = Twitter_homo()
    with open(to_homo_file, 'wb') as f:
        pickle.dump(g, f)
    print(g)

    g = Twitter_hetero()
    with open(to_hetero_file, 'wb') as f:
        pickle.dump(g, f)
    print(g)