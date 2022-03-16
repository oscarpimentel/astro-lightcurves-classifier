from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.rnn.basics as ft_rnn
from fuzzytorch.models.basics import MLP, Linear
import fuzzytorch.models.seq_utils as seq_utils

###################################################################################################################################################

class RNNEncoderS(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()
		### attributes
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()

	def reset(self):
		### input
		nof_bands = len(self.band_names)
		self.x_projection = Linear(self.input_dims+1+nof_bands, self.embd_dims, # input+dtime+bands (i+1+b,f)
			activation='linear',
			bias=False,
			)
		print(f'x_projection={self.x_projection}')

		### rnn
		self.ml_rnn = getattr(ft_rnn, f'ML{self.rnn_cell_name}')(self.embd_dims, self.embd_dims, [self.embd_dims]*(self.layers-1),
			in_dropout=self.dropout['p'],
			dropout=self.dropout['p'],
			)
		print(f'ml_rnn={self.ml_rnn}')

		### seft (and last sequence step)
		self.seft = seq_utils.LinearSEFT(self.embd_dims,
			in_dropout=self.dropout['p'],
			dummy=self.dummy_seft,
			)
		print(f'seft={self.seft}')

		### batch norm
		self.bn = nn.BatchNorm1d(self.embd_dims)
		print(f'bn={self.bn}')
		
	def get_info(self):
		pass

	def get_output_dims(self):
		return self.embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_rnn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		encz_bdict = {}
		s_onehot = tdict[f'input/s_onehot'] # (n,t,d)
		onehot = tdict[f'input/onehot.*'][...,0] # (n,t)
		#rtime = tdict[f'input/rtime.*'][...,0] # (n,t)
		dtime = tdict[f'input/dtime.*'][...,0] # (n,t)
		x = tdict[f'input/x.*'] # (n,t,i)
		#rerror = tdict[f'target/rerror.*'] # (n,t,1)
		#recx = tdict[f'target/recx.*'] # (n,t,1)

		encz = self.x_projection(torch.cat([x, dtime[...,None], s_onehot.float()], dim=-1)) # (n,t,i+1+b)>(n,t,f)
		encz, _ = self.ml_rnn(encz, onehot, **kwargs) # out, (ht, ct)
		encz_bdict[f'encz'] = self.seft(encz, onehot) # (n,t,d)>(n,d)

		### return
		encz_last = encz_bdict[f'encz']
		tdict[f'model/encz_last'] = self.bn(encz_last) # (n,f)>(n,f)
		return tdict

###################################################################################################################################################

class RNNEncoderP(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		### attributes
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()

	def reset(self):
		### input
		nof_bands = len(self.band_names)
		band_embedding_dims = self.embd_dims//nof_bands
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims+1, band_embedding_dims, # input+dtime (i+1,f/b)
			activation='linear',
			bias=False,
			) for b in self.band_names})
		print(f'x_projection={self.x_projection}')

		### rnn
		self.ml_rnn = nn.ModuleDict({b:getattr(ft_rnn, f'ML{self.rnn_cell_name}')(band_embedding_dims, band_embedding_dims, [band_embedding_dims]*(self.layers-1),
			in_dropout=self.dropout['p'],
			dropout=self.dropout['p'],
			) for b in self.band_names})
		print(f'ml_rnn={self.ml_rnn}')

		### seft (and last sequence step)
		self.seft = nn.ModuleDict({b:seq_utils.LinearSEFT(band_embedding_dims,
			in_dropout=self.dropout['p'],
			dummy=self.dummy_seft,
			) for b in self.band_names})
		print(f'seft={self.seft}')

		### multi-band projection
		self.mb_projection = Linear(band_embedding_dims*nof_bands, band_embedding_dims*nof_bands,
			in_dropout=self.dropout['p'],
			activation='linear',
			bias=False,
			)
		print(f'mb_projection={self.mb_projection}')

		### batch norm
		self.bn = nn.BatchNorm1d(self.embd_dims)
		print(f'bn={self.bn}')

	def get_info(self):
		pass

	def get_output_dims(self):
		return self.embd_dims
	
	def get_embd_dims_list(self):
		return {b:self.ml_rnn[b].get_embd_dims_list() for b in self.band_names}

	def forward(self, tdict:dict, **kwargs):
		encz_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (n,t)
			#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (n,t)
			p_dtime = tdict[f'input/dtime.{b}'][...,0] # (n,t)
			p_x = tdict[f'input/x.{b}'] # (n,t,i)
			#p_rerror = tdict[f'target/rerror.{b}'] # (n,t,1)
			#p_recx = tdict[f'target/recx.{b}'] # (n,t,1)

			p_encz = self.x_projection[b](torch.cat([p_x, p_dtime[...,None]], dim=-1)) # (n,t,i+1)>(n,t,f)
			p_encz, _ = self.ml_rnn[b](p_encz, p_onehot, **kwargs) # out, (ht, ct)
			p_encz = self.seft[b](p_encz, p_onehot) # (n,t,d)>(n,d)
			encz_bdict[f'encz.{b}'] = p_encz

		### return
		encz_last = self.mb_projection(torch.cat([encz_bdict[f'encz.{b}'] for b in self.band_names], dim=-1)) # (n,f)>(n,f)
		tdict[f'model/encz_last'] = self.bn(encz_last) # (n,f)>(n,f)
		return tdict