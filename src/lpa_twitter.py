import numpy as np
import networkx as nx
from sklearn.metrics import confusion_matrix
import pickle
import scipy
import random



def load_graph(graph_file, label_rate=0.8):
    with open(graph_file, "rb") as f:
        nx_graph = pickle.load(f)
        nx_graph = nx_graph.to_undirected()


    N = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(nx_graph)
    # S = scipy.sparse.identity(len(nx_graph.nodes())) - N
    S = N
    labels = nx_graph.nodes('labels')
    Y0 = []
    for item in labels:
        Y0.append(item[1][0])
    Y0 = np.array(Y0).reshape([-1, 1])
    truth = Y0.copy()

    # n_nodes = len(nx_graph.nodes())
    # n_train = int(n_nodes * 0.5)
    # n_val = int(n_nodes * 0.1)
    # tmp = [0] * n_train + [1] * n_val + [2] * (n_nodes - n_train - n_val)
    #
    # random.shuffle(tmp)
    # tmp = np.array(tmp)
    #
    # train_idx = np.where(tmp == 0)[0]
    Y0[int(len(nx_graph.nodes()) * label_rate):, :] = 0
    return S, scipy.sparse.csr_matrix(Y0), truth


def lpa(graph_file, label_rate=0.8, gamma=0.5):
    S, Y0, truth = load_graph(graph_file, label_rate)
    Y = Y0
    rounds = 10000
    f1s, fprs, fnrs = [], [], []
    for i in range(rounds):
        Y = gamma * S.dot(Y) + (1 - gamma) * Y0
        tmp_Y = np.where(Y.todense() > 0, 1, 0)
        tn, fp, fn, tp = confusion_matrix(truth, tmp_Y).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        if i >= rounds - 100:
            f1s.append(f1)
            fprs.append(fpr)
            fnrs.append(fnr)

    print('Label rate: {}, f1: {}, fpr: {}, fnr: {}'.format(
        label_rate, round(np.mean(f1s), 5), round(np.mean(fprs), 5), round(np.mean(fnrs), 5)
    ))


if __name__ == '__main__':
    random.seed(0)

    for graph_file in [
        # '../data/nx_graph0930.pkl',
        '../data/nx_graph1031.pkl',
        # '../data/nx_graph1031_wcc1.pkl'
    ]:
        print(graph_file)
        for label_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            lpa(graph_file, label_rate=label_rate, gamma=0.1)