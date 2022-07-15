from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F
import fuzzytorch.models.rnn.basics as ft_rnn
from fuzzytorch.models.basics import MLP, Linear
import fuzzytorch.models.seq_utils as seq_utils

DEC_MLP_NOF_LAYERS = 1
DEC_NOF_LAYERS = 1

###################################################################################################################################################

class LatentGRUDecoderS(nn.Module):
    def __init__(self,
        **kwargs):
        super().__init__()
        ### ATTRIBUTES
        for name, val in kwargs.items():
            setattr(self, name, val)
        self.reset()
        
    def reset(self):
        ### rnn
        nof_bands = len(self.band_names)
        self.ml_rnn = getattr(ft_rnn, f'MLGRU')(1+nof_bands, self.embd_dims, [self.embd_dims]*(DEC_NOF_LAYERS-1), # dtime+bands (1+b,f)
            in_dropout=0.0,
            dropout=self.dropout['p'],
            )
        print(f'ml_rnn={self.ml_rnn}')

        ### mlp
        self.dz_projection = MLP(self.embd_dims, 1, [self.embd_dims]*DEC_MLP_NOF_LAYERS,
            in_dropout=self.dropout['p'],
            dropout=self.dropout['p'],
            activation='relu',
            last_activation='linear',
            )
        print(f'dz_projection={self.dz_projection}')

    def get_output_dims(self):
        return self.dz_projection.get_output_dims()

    def get_embd_dims_list(self):
        return self.ml_rnn.get_embd_dims_list()
        
    def forward(self, tdict:dict, **kwargs):
        s_onehot = tdict[f'input/s_onehot'] # (n,t,b)
        onehot = tdict[f'input/onehot.*'][...,0] # (n,t)
        #rtime = tdict[f'input/rtime.*'][...,0] # (n,t)
        dtime = tdict[f'input/dtime.*'][...,0] # (n,t)
        #x = tdict[f'input/x.*'] # (n,t,i)
        #rerror = tdict[f'target/rerror.*'] # (n,t,1)
        #rx = tdict[f'target/rec_x.*'] # (n,t,1)
        encz_last = tdict[f'model/encz_last'][None,...] # (n,f)>(1,n,f)

        decz = torch.cat([dtime[...,None], s_onehot.float()], dim=-1) # (n,t,1+b)
        decz, _ = self.ml_rnn(decz, onehot, # out, (ht,ct)
            h0=encz_last,
            )
        decx = self.dz_projection(decz) # (n,t,f)>(n,t,1)
        for kb,b in enumerate(self.band_names):
            p_decx = seq_utils.serial_to_parallel(decx, s_onehot[...,kb])
            tdict[f'model/decx.{b}'] = p_decx

        return tdict

###################################################################################################################################################

class LatentGRUDecoderP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        ### ATTRIBUTES
        for name, val in kwargs.items():
            setattr(self, name, val)
        self.reset()
    
    def reset(self):
        ### rnn
        self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'MLGRU')(1, self.embd_dims, [self.embd_dims]*(DEC_NOF_LAYERS-1), # dtime (1,f)
            in_dropout=0.0,
            dropout=self.dropout['p'],
            ) for b in self.band_names})
        print(f'ml_rnn={self.ml_rnn}')

        ### mlp
        self.dz_projection = nn.ModuleDict({b:MLP(self.embd_dims, 1, [self.embd_dims]*DEC_MLP_NOF_LAYERS,
            in_dropout=self.dropout['p'],
            dropout=self.dropout['p'],
            activation='relu',
            last_activation='linear',
            ) for b in self.band_names})
        print(f'dz_projection={self.dz_projection}')

    def get_output_dims(self):
        return self.dz_projection.get_output_dims()

    def get_embd_dims_list(self):
        return {b:self.ml_rnn[b].get_embd_dims_list() for b in self.band_names}
        
    def forward(self, tdict:dict, **kwargs):
        for kb,b in enumerate(self.band_names):
            p_onehot = tdict[f'input/onehot.{b}'][...,0] # (n,t)
            #p_rtime = tdict[f'input/rtime.{b}'][...,0] # (n,t)
            p_dtime = tdict[f'input/dtime.{b}'][...,0] if self.preserved_band=='.' else tdict[f'input/dtime_pb.{b}'][...,0] # (n,t)
            #p_x = tdict[f'input/x.{b}'] # (b,t,i)
            #p_rerror = tdict[f'target/rerror.{b}'] # (n,t,1)
            #p_rx = tdict[f'target/rec_x.{b}'] # (n,t,1)
            encz_last = tdict[f'model/encz_last'][None,...] # (n,f)>(1,n,f)

            p_decz = p_dtime[...,None] # (n,t)>(n,t,1)
            p_decz, _ = self.ml_rnn[b](p_decz, p_onehot, # out, (ht,ct)
                h0=encz_last,
                )
            p_decx = self.dz_projection[b](p_decz) # (n,t,f)>(n,t,1)
            tdict[f'model/decx.{b}'] = p_decx

        return tdict