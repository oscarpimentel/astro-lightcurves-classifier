from __future__ import print_function
from __future__ import division
from . import _C

import math
import torch
import torch.nn.functional as F
from fuzzytorch.losses import FTLoss
import fuzzytorch.models.seq_utils as seq_utils
import math

XENTROPY_K = _C.XENTROPY_K
MSE_K = _C.MSE_K
REC_LOSS_EPS = _C.REC_LOSS_EPS
REC_LOSS_K = _C.REC_LOSS_K

###################################################################################################################################################

class LCWMSEReconstruction(FTLoss):
	def __init__(self, name, weight_key,
		band_names=None,
		preserved_band=None,
		**kwargs):
		super().__init__(name, weight_key)
		self.band_names = band_names
		self.preserved_band = preserved_band

	def compute_loss(self, tdict,
		**kwargs):
		wmse_loss_bdict = {}
		for kb,b in enumerate(self.band_names):
			p_onehot = tdict[f'input/onehot.{b}'][...,0] # (n,t)
			#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (n,t)
			#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (n,t)
			#p_x = tdict[f'input/x.{b}'] # (n,t,i)
			p_rerror = tdict[f'target/rerror.{b}'] # (n,t,1)
			p_recx = tdict[f'target/recx.{b}'] # (n,t,1)
			p_decx = tdict[f'model/decx.{b}'] # (n,t,1)

			p_wmse_loss = (p_recx-p_decx)**2/(REC_LOSS_K*(p_rerror**2)+REC_LOSS_EPS) # (n,t,1)
			p_wmse_loss = seq_utils.seq_avg_pooling(p_wmse_loss, p_onehot)[...,0] # (n,t,1)>(n,t)>(n)
			p_wmse_loss = p_wmse_loss*0 if not self.preserved_band=='.' and not b==self.preserved_band else p_wmse_loss # for ablation
			wmse_loss_bdict[b] = p_wmse_loss

		wmse_loss = torch.cat([wmse_loss_bdict[b][...,None] for b in self.band_names], axis=-1).mean(dim=-1) # (n,d)>(n)
		return wmse_loss # (n)

###################################################################################################################################################

class LCBinXEntropy(FTLoss):
	def __init__(self, name, weight_key,
		class_names=None,
		target_is_onehot:bool=False,
		target_y_key='target/y',
		pred_y_key='model/y',
		**kwargs):
		super().__init__(name, weight_key)
		self.class_names = class_names
		self.target_is_onehot = target_is_onehot
		self.target_y_key = target_y_key
		self.pred_y_key = pred_y_key
		self.reset()

	def reset(self):
		self.bin_loss = torch.nn.BCELoss(reduction='none')

	def get_onehot(self, y):
		class_count = torch.max(y)[0] if self.class_names is None else len(self.class_names)
		return torch.eye(class_count, device=y.device)[y,:]

	def compute_loss(self, tdict,
		**kwargs):
		y_target = tdict[self.target_y_key] # (n)
		y_pred = tdict[self.pred_y_key] # (n,c)

		y_target = y_target if self.target_is_onehot else self.get_onehot(y_target) # (n,c)
		y_pred_p = torch.softmax(y_pred, dim=-1) # (n,c)
		binxentropy_loss = self.bin_loss(y_pred_p, y_target) # (n,c)
		binxentropy_loss = binxentropy_loss.mean(dim=-1) # (n,c)>(n)
		return binxentropy_loss # (n)

###################################################################################################################################################

class LCCompleteLoss(FTLoss):
	def __init__(self, name, weight_key,
		band_names=None,
		preserved_band=None,
		class_names=None,
		target_is_onehot:bool=False,
		target_y_key='target/y',
		pred_y_key='model/y',
		binxentropy_k=XENTROPY_K,
		wmse_k=WMSE_K,
		**kwargs):
		super().__init__(name, weight_key)
		self.band_names = band_names
		self.preserved_band = preserved_band
		self.class_names = class_names
		self.target_is_onehot = target_is_onehot
		self.target_y_key = target_y_key
		self.pred_y_key = pred_y_key
		self.binxentropy_k = binxentropy_k
		self.wmse_k = wmse_k
		self.reset()

	def reset(self):
		self.binxentropy = LCBinXEntropy('', None,
			self.class_names,
			self.target_is_onehot,
			self.target_y_key,
			self.pred_y_key,
			)
		self.wmse = LCWMSEReconstruction('', None,
			self.band_names,
			self.preserved_band,
			)

	def compute_loss(self, tdict,
		**kwargs):
		binxentropy_loss = self.binxentropy.compute_loss(tdict, **kwargs)*self.binxentropy_k # (n)
		wmse_loss = self.wmse.compute_loss(tdict, **kwargs)*self.wmse_k # (n)
		d = {
			'_loss':binxentropy_loss+wmse_loss, # (n)
			'binxentropy':binxentropy_loss, # (n)
			'wmse':wmse_loss,
			}
		return d
