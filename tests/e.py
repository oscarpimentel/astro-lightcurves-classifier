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

x = torch.rand(1,10,1)
onehot = torch.zeros_like(x)[...,0]
new_onehot = onehot.clone().detach()
new_onehot[:,:5] = 1
new_x = seq_utils.seq_min_max_norm(x, new_onehot.bool())
print(x)
print('onehot', onehot)
print('new_onehot', new_onehot)
print(new_x)

model = nn.Linear(5, 5, bias=False)
state_dict_copy = copy(model.state_dict())
state_dict_deepcopy = deepcopy(model.state_dict())
print(f'state_dict={model.state_dict()}')
model.reset_parameters()
print('change dict')
print(f'state_dict={model.state_dict()}')
print(f'state_dict_copy={state_dict_copy}')
print(f'state_dict_deepcopy={state_dict_deepcopy}')
for k in model.state_dict().keys():
    print(model.state_dict()[k].equal(state_dict_copy[k]))
    print(model.state_dict()[k].equal(state_dict_deepcopy[k]))