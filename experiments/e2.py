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
import torch.nn as nn
import numpy as np
from fuzzytorch.models.basics import Linear

split_out = 2
n = 2
t = 5
f = 4
te_code = np.random.rand(n, t, f)
linear = Linear(f, f,
	bias=False,
	split_out=2,
	)
print('te_code', te_code)

### torch
a, b = linear(torch.Tensor(te_code))
print('a', a)
print('b', b)

### numpy
weight = linear.linear.weight.cpu().detach().numpy()
weight = weight.T
a, b = np.split(te_code@weight, split_out, axis=-1)
print('a', a)
print('b', b)