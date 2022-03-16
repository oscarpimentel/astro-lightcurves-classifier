#!/usr/bin/env python3
# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import numpy as np
from copy import copy, deepcopy

import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-torch') # or just install the module
sys.path.append('../../fuzzy-tools') # or just install the module

import torch
import fuzzytorch.models.seq_utils as seq_utils

embedding_dim = 128
layer_norm = nn.LayerNorm([embedding_dim])

n = 3
x1 = torch.rand((n, 10, embedding_dim))
x2 = torch.zeros((n, 20, embedding_dim))

x = torch.cat([x1, x2], dim=1)
print('x', x[0,:,0])
x_norm = layer_norm(x)
print('x_norm', x_norm[0,:,0])

x = torch.cat([x1, x2, x2, x2, x1*100], dim=1)
#x[:,:,-1] += 100
# x[:,-1,:] += 100
x[-1,:,:] += 100
print('x', x[0,:,0])
x_norm = layer_norm(x)
print('x_norm', x_norm[0,:,0])
