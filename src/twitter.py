#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
import time
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from torch import nn

from models_ours import GAT as GAT_ours
from models_bot import GAT as GAT_bot
from models_ngnn import GAT as GAT_ngnn
from utils import BatchSampler, DataLoaderWrapper
from twitter_dataset import Twitter_homo, Twitter_hetero

device = None
n_node_feats = None
n_edge_feats = 2
n_classes = 1


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def score(y_pred, y_true):
    # converting to torch.Tensor to numpy on cpu
    if torch is not None and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if torch is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    ## check type
    if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
        raise RuntimeError('Arguments to score need to be either numpy ndarray or torch tensor')

    if not y_true.shape == y_pred.shape:
        raise RuntimeError('Shape of y_true and y_pred must be the same')

    if not y_true.shape[1] == 1:
        raise RuntimeError('y_true and y_pred must be array-like of shape (n_samples, 1)')

    tn, fp, fn, tp = confusion_matrix(y_true[:, 0], y_pred[:, 0]).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, fpr, fnr
    # try:
    #     return roc_auc_score(y_true[:, 0], y_pred[:, 0])
    # except ValueError:
    #     return f1_score(y_true[:, 0], y_pred[:, 0])


def load_data(dgl_file):

    with open(dgl_file, "rb") as f:
        graph = pickle.load(f)[0]
        labels = graph.ndata['labels']

        n_nodes = graph.num_nodes()
        n_train = int(n_nodes * 0.8)
        n_val = int(n_nodes * 0.1)
        tmp = [0] * n_train + [1] * n_val + [2] * (n_nodes - n_train - n_val)
        random.shuffle(tmp)
        tmp = torch.tensor(tmp)

        train_idx = (tmp == 0).nonzero(as_tuple=True)[0]
        val_idx = (tmp == 1).nonzero(as_tuple=True)[0]
        test_idx = (tmp == 2).nonzero(as_tuple=True)[0]

    return graph, labels, train_idx, val_idx, test_idx


def preprocess(graph, labels, train_idx, val_idx, test_idx, args):
    global n_node_feats

    if args.model == 'bot':
        # set float type
        graph.ndata['feat'] = graph.ndata['feat'].float()
        graph.edata['feat'] = graph.edata['feat'].float()

        # The sum of the weights of adjacent edges is used as node features.
        graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat_"))
        graph.ndata['feat'] = torch.cat([graph.ndata['feat'], graph.ndata['feat_']], dim=1)
        n_node_feats = graph.ndata["feat"].shape[-1]

        # save memory
        graph.ndata.pop('feat_')

        # normalization
        graph.ndata['feat'] = torch.from_numpy(StandardScaler().fit_transform(graph.ndata['feat'].numpy()))
        graph.edata['feat'] = torch.from_numpy(StandardScaler().fit_transform(graph.edata['feat'].numpy()))
        # train_ndata = graph.ndata['feat'][train_idx]
        # test_ndata = graph.ndata['feat'][torch.cat([val_idx, test_idx])]
        # scaler = StandardScaler()
        # train_ndata = scaler.fit_transform(train_ndata)
        # test_ndata = scaler.transform(test_ndata)
        # graph.ndata['feat'][train_idx] = train_ndata
        # graph.ndata['feat'][torch.cat([val_idx, test_idx])] = test_ndata

        # Only the labels in the training set are used as features, while others are filled with zeros.
        graph.ndata["train_labels_onehot"] = torch.zeros(graph.number_of_nodes(), n_classes, dtype=torch.float)
        graph.ndata["train_labels_onehot"][train_idx, 0] = labels[train_idx, 0].float()
        graph.ndata['is_train'] = torch.zeros([graph.num_nodes()]).int()
        graph.ndata['is_train'][train_idx] += 1

        graph.ndata["deg"] = graph.out_degrees().float().clamp(min=1)

    elif args.model == 'ours':
        # set float type
        graph.ndata['feat'] = graph.ndata['feat'].float()
        graph.edges['to'].data['feat'] = graph.edges['to'].data['feat'].float()
        graph.edges['_to'].data['feat'] = graph.edges['_to'].data['feat'].float()

        # The sum of the weights of adjacent edges is used as node features.
        graph.multi_update_all({
            'to': (fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat_1")),
            '_to': (fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat_2"))
        }, 'sum')
        graph.ndata['feat'] = torch.cat([graph.ndata['feat'], graph.ndata['feat_1'], graph.ndata['feat_2']], dim=1)
        n_node_feats = graph.ndata["feat"].shape[-1]

        # save memory
        graph.ndata.pop('feat_1')
        graph.ndata.pop('feat_2')

        # normalization
        graph.ndata['feat'] = torch.from_numpy(StandardScaler().fit_transform(graph.ndata['feat'].numpy()))
        graph.edges['to'].data['feat'] = torch.from_numpy(StandardScaler().fit_transform(graph.edges['to'].data['feat'].numpy()))
        graph.edges['_to'].data['feat'] = torch.from_numpy(StandardScaler().fit_transform(graph.edges['_to'].data['feat'].numpy()))

        # Only the labels in the training set are used as features, while others are filled with zeros.
        graph.ndata["train_labels_onehot"] = torch.zeros(graph.number_of_nodes(), n_classes, dtype=torch.float)
        graph.ndata["train_labels_onehot"][train_idx, 0] = labels[train_idx, 0].float()
        graph.ndata['is_train'] = torch.zeros([graph.num_nodes()]).int()
        graph.ndata['is_train'][train_idx] += 1

        graph.ndata["deg"] = graph.out_degrees(etype='to').float().clamp(min=1)

    else:
        assert False

    graph.create_formats_()

    return graph, labels


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    if args.model == 'bot':
        model = GAT_bot(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=16,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst,
        )
    elif args.model == 'ngnn':
        model = GAT_ngnn(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=0,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst,
        )
    elif args.model == 'ours':
        model = GAT_ours(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=16,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst,
        )
    else:
        assert False

    return model


def add_labels(graph, idx, label_rate):
    train_idx = (graph.srcdata["is_train"][idx] == 1).nonzero(as_tuple=True)[0]
    max_label_rate = train_idx.size(0) / len(graph.srcnodes())
    if max_label_rate > label_rate:
        perm = torch.randperm(train_idx.size(0))
        idx = idx[train_idx[perm[:int(len(graph.srcnodes()) * label_rate)]]]
    feat = graph.srcdata["feat"]
    train_labels_onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    train_labels_onehot[idx] = graph.srcdata["train_labels_onehot"][idx]
    graph.srcdata["feat"] = torch.cat([feat, train_labels_onehot], dim=-1)


def train(args, model, dataloader, _labels, _train_idx, criterion, optimizer):
    model.train()

    loss_sum, total = 0, 0

    for input_nodes, output_nodes, subgraphs in dataloader:
        subgraphs = [b.to(device) for b in subgraphs]
        new_train_idx = torch.arange(len(output_nodes), device=device)

        if args.use_labels:
            train_labels_idx = torch.arange(len(output_nodes), len(input_nodes), device=device)
            train_pred_idx = new_train_idx

            add_labels(subgraphs[0], train_labels_idx, args.label_rate)
        else:
            train_pred_idx = new_train_idx

        pred = model(subgraphs)
        loss = criterion(pred[train_pred_idx], subgraphs[-1].dstdata["labels"][train_pred_idx].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = len(train_pred_idx)
        loss_sum += loss.item() * count
        total += count

        # torch.cuda.empty_cache()

    return loss_sum / total


@torch.no_grad()
def evaluate(args, model, dataloader, labels, train_idx, val_idx, test_idx, criterion):
    model.eval()

    preds = torch.zeros(labels.shape).to(device)

    # Due to the memory capacity constraints, we use sampling for inference and calculate the average of the predictions 'eval_times' times.
    eval_times = 1

    for _ in range(eval_times):
        for input_nodes, output_nodes, subgraphs in dataloader:
            subgraphs = [b.to(device) for b in subgraphs]
            new_train_idx = torch.tensor(list(range(len(input_nodes))))

            if args.use_labels:
                add_labels(subgraphs[0], new_train_idx, args.label_rate)

            pred = model(subgraphs)
            preds[output_nodes] += pred

            # torch.cuda.empty_cache()

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()

    threshold = 0.5
    train_preds = torch.gt(torch.sigmoid(preds[train_idx]), torch.zeros_like(preds[train_idx]) + threshold).int()
    val_preds = torch.gt(torch.sigmoid(preds[val_idx]), torch.zeros_like(preds[val_idx]) + threshold).int()
    test_preds = torch.gt(torch.sigmoid(preds[test_idx]), torch.zeros_like(preds[test_idx]) + threshold).int()

    train_f1, train_fpr, train_fnr = score(train_preds, labels[train_idx])
    val_f1, val_fpr, val_fnr = score(val_preds, labels[val_idx])
    test_f1, test_fpr, test_fnr = score(test_preds, labels[test_idx])
    return (
        train_f1, train_fpr, train_fnr,
        val_f1, val_fpr, val_fnr,
        test_f1, test_fpr, test_fnr,
        train_loss, val_loss, test_loss,
        preds,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, n_running):

    train_batch_size = (len(train_idx) + 9) // 10
    # train_sampler = MultiLayerNeighborSampler([32 for _ in range(args.n_layers)])
    train_sampler = MultiLayerFullNeighborSampler(args.n_layers)
    train_dataloader = DataLoaderWrapper(
        NodeDataLoader(
            graph.cpu(),
            train_idx.cpu(),
            train_sampler,
            batch_sampler=BatchSampler(len(train_idx), batch_size=train_batch_size)
        )
    )

    eval_batch_size = len(train_idx) + len(val_idx) + len(test_idx)
    # eval_sampler = MultiLayerNeighborSampler([100 for _ in range(args.n_layers)])
    eval_sampler = MultiLayerFullNeighborSampler(args.n_layers)
    eval_dataloader = DataLoaderWrapper(
        NodeDataLoader(
            graph.cpu(),
            torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
            eval_sampler,
            batch_sampler=BatchSampler(graph.number_of_nodes(), batch_size=eval_batch_size)
        )
    )

    criterion = nn.BCEWithLogitsLoss()

    model = gen_model(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)

    total_time = 0
    val_f1, best_val_f1, best_val_fpr, best_val_fnr = -1, -1, -1, -1
    final_test_f1, final_test_fpr, final_test_fnr = -1, -1, -1

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        loss = train(args, model, train_dataloader, labels, train_idx, criterion, optimizer)

        toc = time.time()
        total_time += toc - tic

        if epoch == args.n_epochs or epoch % args.eval_every == 0 or epoch % args.log_every == 0:
            train_f1, train_fpr, train_fnr, \
            val_f1, val_fpr, val_fnr, \
            test_f1, test_fpr, test_fnr,\
            train_loss, val_loss, test_loss, pred = evaluate(
                args, model, eval_dataloader, labels, train_idx, val_idx, test_idx, criterion
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_fpr = val_fpr
                best_val_fnr = val_fnr

                final_test_f1 = test_f1
                final_test_fpr = test_fpr
                final_test_fnr = test_fnr

            if epoch % args.log_every == 0:
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}s"
                )
                print(
                    f"Avg loss [current train epoch]: {loss:.4f}\n"
                    f"Loss [Train/Val/Test]: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"F1 [Train/Val/Test/Best_val/Final_test]: {train_f1:.4f}/{val_f1:.4f}/{test_f1:.4f}/{best_val_f1:.4f}/{final_test_f1:.4f}\n"
                    f"FPR [Train/Val/Test/Best_val/Final_test]: {train_fpr:.4f}/{val_fpr:.4f}/{test_fpr:.4f}/{best_val_fpr:.4f}/{final_test_fpr:.4f}\n"
                    f"FNR [Train/Val/Test/Best_val/Final_test]: {train_fnr:.4f}/{val_fnr:.4f}/{test_fnr:.4f}/{best_val_fnr:.4f}/{final_test_fnr:.4f}\n"
                )

        lr_scheduler.step(val_f1)

    print("*" * 50)
    print(f"Final test F1: {final_test_f1} | Final test FPR: {final_test_fpr} | Final test FNR: {final_test_fnr}")
    print("*" * 50)

    return final_test_f1, final_test_fpr, final_test_fnr  # best_val_score, final_test_score


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-proteins", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides '--gpu'.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=123, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=100, help="number of epochs")
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--n-heads", type=int, default=4, help="number of heads")
    argparser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=4, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=64, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=5, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")

    argparser.add_argument("--model", type=str, choices=["bot", "ours", "ngnn"], default="ours", help="model")
    argparser.add_argument("--data", type=str, choices=["lscc", "lwcc"], default="lscc", help="model")
    '''new args'''
    argparser.add_argument("--label-rate", type=float, default=0.5, help="label rate")
    args = argparser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    print("Loading data")
    if args.model == 'bot':
        dataset = '../data/dgl1031_homo'
    elif args.model == 'ours':
        dataset = '../data/dgl1031_hetero'
    else:
        assert False
    if args.data == 'lwcc':
        dataset += '_wcc1'
    dataset += '.pkl'
    graph, labels, train_idx, val_idx, test_idx = load_data(dataset)
    print("Preprocessing")
    graph, labels = preprocess(graph, labels, train_idx, val_idx, test_idx, args)

    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    test_f1s, test_fprs, test_fnrs = [], [], []

    for i in range(args.n_runs):
        print("Running", i)
        set_seed(args.seed + i)
        test_f1, test_fpr, test_fnr = run(args, graph, labels, train_idx, val_idx, test_idx, i + 1)
        test_f1s.append(test_f1)
        test_fprs.append(test_fpr)
        test_fnrs.append(test_fnr)

    print(" ".join(sys.argv))
    print(args)
    print(f"Runned {args.n_runs} times")
    print("Test F1s:", test_f1s)
    print("Test FPRs:", test_fprs)
    print("Test FNRs:", test_fnrs)
    print(f"Average F1: {np.mean(test_f1s)} ± {np.std(test_f1s)}")
    print(f"Average FPR: {np.mean(test_fprs)} ± {np.std(test_fprs)}")
    print(f"Average FNR: {np.mean(test_fnrs)} ± {np.std(test_fnrs)}")
    print(f"Number of params: {count_parameters(args)}")

if __name__ == "__main__":
    main()
