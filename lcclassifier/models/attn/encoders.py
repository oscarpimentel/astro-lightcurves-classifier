from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F 
import fuzzytorch.models.attn.basics as ft_attn
from fuzzytorch.models.basics import MLP, Linear
import fuzzytorch.models.seq_utils as seq_utils

ATTN_DROPOUT = _C.ATTN_DROPOUT

###################################################################################################################################################

class TimeModSelfAttnEncoderS(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()
		### attributes
		self.add_extra_return = False
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()

	def reset(self):
		### input
		nof_bands = len(self.band_names)
		self.x_projection = Linear(self.input_dims+nof_bands, self.embd_dims, # input+bands (i+b,f)
			activation='linear',
			bias=False,
			)
		print(f'x_projection={self.x_projection}')
		
		### attn
		self.ml_attn = ft_attn.MLTimeSelfAttn(self.embd_dims, self.embd_dims, [self.embd_dims]*(self.layers-1), self.te_features, self.max_period,
			kernel_size=self.kernel_size,
			time_noise_window=self.time_noise_window,
			num_heads=self.heads*nof_bands,
			in_dropout=self.dropout['p'],
			residual_dropout=self.dropout['r'],
			dropout=self.dropout['p'],
			attn_dropout=ATTN_DROPOUT,
			)
		print(f'ml_attn={self.ml_attn}')

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
		return {
			'ml_attn':self.ml_attn.get_info(),
			}

	def get_output_dims(self):
		return self.embd_dims
	
	def get_embd_dims_list(self):
		return self.ml_attn.get_embd_dims_list()

	def forward(self, tdict:dict, **kwargs):
		encz_bdict = {}
		s_onehot = tdict[f'input/s_onehot'] # (n,t,b)
		onehot = tdict[f'input/onehot.*'][...,0] # (n,t)
		rtime = tdict[f'input/rtime.*'][...,0] # (n,t)
		#dtime = tdict[f'input/dtime.*'][...,0] # (n,t)
		x = tdict[f'input/x.*'] # (n,t,i)
		#rerror = tdict[f'target/rerror.*'] # (n,t,1)
		#recx = tdict[f'target/recx.*'] # (n,t,1)

		encz = self.x_projection(torch.cat([x, s_onehot.float()], dim=-1)) # (n,t,i+b)>(n,t,f)
		encz, scores = self.ml_attn(encz, onehot, rtime, return_only_actual_scores=True)
		if self.add_extra_return:
			tdict[f'model/attn_scores/encz'] = scores # (n,l,h,qt)
		encz_bdict[f'encz'] = self.seft(encz, onehot) # (n,t,f)>(n,f)

		### return
		encz_last = encz_bdict[f'encz']
		tdict[f'model/encz_last'] = self.bn(encz_last) # (n,f)>(n,f)
		return tdict

###################################################################################################################################################

class TimeModSelfAttnEncoderP(nn.Module):
	def __init__(self,
		**kwargs):
		super().__init__()
		### attributes
		self.add_extra_return = False
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()

	def reset(self):
		### input
		nof_bands = len(self.band_names)
		band_embedding_dims = self.embd_dims//nof_bands
		self.x_projection = nn.ModuleDict({b:Linear(self.input_dims, band_embedding_dims, # input (i,f/b)
			activation='linear',
			bias=False,
			) for b in self.band_names})
		print(f'x_projection={self.x_projection}')

		### attn
		self.ml_attn = nn.ModuleDict({b:ft_attn.MLTimeSelfAttn(band_embedding_dims, band_embedding_dims, [band_embedding_dims]*(self.layers-1), self.te_features, self.max_period,
			kernel_size=self.kernel_size,
			time_noise_window=self.time_noise_window,
			num_heads=self.heads,
			in_dropout=self.dropout['p'],
			residual_dropout=self.dropout['r'],
			dropout=self.dropout['p'],
			attn_dropout=ATTN_DROPOUT,
			) for b in self.band_names})
		print(f'ml_attn={self.ml_attn}')

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
		d = {}
		for kb,b in enumerate(self.band_names):
			d[f'ml_attn.{b}'] = self.ml_attn[b].get_info()
		return d

	def get_output_dims(self):
		return self.embd_dims
	
	def get_embd_dims_list(self):
		return {b:self.ml_attn[b].get_embd_dims_list() for b in self.band_names}

	def forward(self, tdict:dict, **kwargs):
		encz_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (n,t)
			p_rtime = tdict[f'input/rtime.{b}'][...,0] if self.preserved_band=='.' else tdict[f'input/rtime_pb.{b}'][...,0] # (n,t)
			#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (n,t)
			p_x = tdict[f'input/x.{b}'] # (n,t,i)
			#p_rerror = tdict[f'target/rerror.{b}'] # (n,t,1)
			#p_recx = tdict[f'target/recx.{b}'] # (n,t,1)

			p_encz = self.x_projection[b](p_x) # (n,t,i)>(n,t,f)
			p_encz, p_scores = self.ml_attn[b](p_encz, p_onehot, p_rtime, return_only_actual_scores=True)
			if self.add_extra_return:
				tdict[f'model/attn_scores/encz.{b}'] = p_scores # (n,l,h,qt)
			p_encz = self.seft[b](p_encz, p_onehot) # (n,t,f)>(n,f)
			p_encz = p_encz*0 if not self.preserved_band=='.' and not b==self.preserved_band else p_encz # ablation
			encz_bdict[f'encz.{b}'] = p_encz
		
		### return
		encz_last = self.mb_projection(torch.cat([encz_bdict[f'encz.{b}'] for b in self.band_names], dim=-1)) # (n,f)>(n,f)
		tdict[f'model/encz_last'] = self.bn(encz_last) # (n,f)>(n,f)
		return tdict