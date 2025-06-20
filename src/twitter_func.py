#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import time
from dgl.data import DGLDataset
import pickle


def read_interaction(date):
    df = pd.read_csv(
        '../data/T_interaction{}.csv'.format(date), sep=',', header=0,
        usecols=['user1', 'user2', 'note'],
        dtype={'user1': str, 'user2': str, 'note': str}
    )
    df.dropna(axis=0, how='any', inplace=True)
    idx_mapping = {'mention': 0, 'replied_to': 0, 'retweeted': 1, 'quoted': 1}
    df['note'] = df['note'].map(lambda x: idx_mapping[x])
    return df


def read_user_label(date):
    df = pd.read_csv(
        '../data/T_userlabel{}.csv'.format(date), sep=',', header=0,
        usecols=['user', 'label'],
        dtype={'user': str, 'label': str}
    )
    df.dropna(axis=0, how='any', inplace=True)
    label_mapping = {'anti': 0, 'pro': 1}
    df['label'] = df['label'].map(lambda x: label_mapping[x])
    user_labels = df.set_index('user')['label'].to_dict()
    return user_labels


def create_nx_graph(date):

    df = read_interaction(date)
    origin_graph = nx.DiGraph()
    df_cleaned = df.drop_duplicates(subset=['user1', 'user2'], keep='first', ignore_index=True)

    # 2D features: mention+reply, retweeted+quoted
    edge_dict = defaultdict(lambda: [0, 0])
    sources = df_cleaned['user1'].values
    targets = df_cleaned['user2'].values
    origin_graph.add_weighted_edges_from([(s, t, edge_dict[(s, t)]) for s, t in zip(sources, targets)], 'feat')

    user_labels = read_user_label(date)
    user_labels = dict([(k, [v]) for k, v in user_labels.items()])

    '''largest_scc: formal'''
    # largest_scc = max(nx.strongly_connected_components(origin_graph), key=len)
    # graph = graph.subgraph(largest_scc)

    # ccs = [c for c in sorted(nx.strongly_connected_components(origin_graph), key=len, reverse=True)]
    ccs = [c for c in sorted(nx.weakly_connected_components(origin_graph), key=len, reverse=True)]
    cc = ccs[0]

    '''experiments'''
    node_set = cc

    '''visualization'''
    # node_set = set()
    # found_anti = 0
    # for node in cc:
    #     if user_labels[node][0] == 0:
    #         node_set.add(node)
    #         found_anti += 1
    #     if found_anti >= 5:
    #         break
    # for j in range(2):
    #     node_set_ = list(node_set).copy()
    #     for node in node_set_:
    #         neighbors = origin_graph.neighbors(node)
    #         node_set.update(neighbors)

    graph = origin_graph.subgraph(node_set)

    def apply_func(x):
        if x['user1'] in node_set and x['user2'] in node_set:
            edge_dict[(x['user1'], x['user2'])][x['note']] += 1
    df.apply(func=apply_func, axis=1)

    '''reshape labels'''
    nx.set_node_attributes(graph, user_labels, name='labels')

    '''centrality'''
    # centrality = [
    #     nx.in_degree_centrality,
    #     nx.out_degree_centrality,
    #     nx.degree_centrality,
    #     # # nx.eigenvector_centrality,
    #     # # nx.closeness_centrality,
    #     # # nx.current_flow_closeness_centrality,
    #     # nx.betweenness_centrality,
    #     # # nx.current_flow_betweenness_centrality,
    #     # # nx.communicability_betweenness_centrality,
    #     # nx.load_centrality,
    #     # # nx.subgraph_centrality,
    #     # nx.harmonic_centrality,
    #     # # nx.global_reaching_centrality,
    #     # # nx.percolation_centrality,
    #     # # nx.second_order_centrality
    # ]
    # measures = []
    # for i in range(len(centrality)):
    #     try:
    #         measure = centrality[i](graph)
    #         measures.append(measure)
    #     except Exception as e:
    #         print('Error in Centrality {}'.format(i + 1))
    #         print(e)

    '''degree'''
    measures = [dict(graph.in_degree), dict(graph.out_degree), dict(graph.degree)]

    node_attributes = {}
    for node in node_set:
        node_attributes[node] = [measure[node] for measure in measures]
    nx.set_node_attributes(graph, node_attributes, name='feat')

    node_ids = list(graph.nodes)
    indices = [i for i in range(len(node_ids))]
    mapping = defaultdict(lambda: -1)
    mapping.update(dict(zip(node_ids, indices)))
    graph = nx.relabel_nodes(graph, mapping=mapping)

    with open('../data/nx_graph{}_wcc1.pkl'.format(date), 'wb') as f:
        pickle.dump(graph, f)
    return graph


if __name__ == '__main__':

    nx_graph = create_nx_graph('1031')
    # nx_graph = create_nx_graph('1231')



