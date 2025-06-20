#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class DataLoaderWrapper(object):
    def __init__(self, dataloader):
        self.iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except Exception:
            raise StopIteration() from None


class BatchSampler(object):
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            shuf = torch.randperm(self.n).split(self.batch_size)
            for shuf_batch in shuf:
                yield shuf_batch
            yield None


def __init_weight(feature, bn_eps, bn_momentum, conv_init, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.uniform_(m.weight, 0.0, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            stdv = 1. / np.sqrt(m.weight.data.size(1))
            m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -1.0, 1.0)
        else:
            raise RuntimeError


def init_business_weight(
        module_list, bn_eps=1e-5, bn_momentum=0.1, conv_init=nn.init.kaiming_normal_, **kwargs
):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, bn_eps, bn_momentum, conv_init, **kwargs)
    else:
        __init_weight(module_list, bn_eps, bn_momentum, conv_init, **kwargs)