from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
from fuzzytorch.models.basics import MLP, Linear

###################################################################################################################################################

class SimpleClassifier(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		### ATTRIBUTES
		for name, val in kwargs.items():
			setattr(self, name, val)
		self.reset()

	def reset(self):
		self.classifier_mlp_ft = MLP(self.input_dims, self.output_dims, [self.input_dims*1]*self.layers,
			activation='relu',
			last_activation='linear',
			in_dropout=self.dropout['p'],
			dropout=self.dropout['p'],
			)
		print('classifier_mlp_ft:', self.classifier_mlp_ft)
		self.reset_parameters()

	def reset_parameters(self):
		self.classifier_mlp_ft.reset_parameters()

	def get_output_dims(self):
		return self.output_dims

	def forward(self, tdict:dict, **kwargs):
		encz_last = tdict[f'model/encz_last'] # (n,f)
		
		y, encz_pre_last = self.classifier_mlp_ft(encz_last, returns_pre_last=True)
		tdict[f'model/y'] = y # (n,c)
		tdict[f'model/encz_pre_last'] = encz_pre_last # (n,f)
		return tdict