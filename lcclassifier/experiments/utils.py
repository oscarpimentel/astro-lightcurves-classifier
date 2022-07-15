from __future__ import print_function
from __future__ import division
from . import _C

import torch

EPS = _C.EPS

###################################################################################################################################################

def check_attn_scores(attn_scores,
    eps=EPS,
    ):
    # print(b, p_attn_scores.shape, p_onehot[0], p_attn_scores[0,0,:], torch.sum(p_attn_scores[0,0,:], dim=-1), torch.sum(p_attn_scores[0,0,:]>1e-32, dim=-1))
    sum_attn_scores = torch.sum(attn_scores, dim=-1) # (n,h,qt)>(n,h)
    return torch.all((torch.abs(sum_attn_scores-1)<eps)|(sum_attn_scores==0))