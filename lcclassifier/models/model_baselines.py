from __future__ import print_function
from __future__ import division
from . import _C

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzytorch.utils import get_model_name, print_tdict
from copy import copy, deepcopy
import torch.autograd.profiler as profiler
from .rnn import encoders as rnn_encoders
from .attn import encoders as attn_encoders
from .rnn import extra_encoders as rnn_extra_encoders
from .attn import extra_encoders as attn_extra_encoders
from .rnn import decoders as rnn_decoders
from . import classifiers as mclass

DECODER_CLASS = rnn_decoders.LatentGRUDecoderP # LatentGRUDecoderP LatentGRUDecoderS
CLASSIFIER_CLASS = mclass.SimpleClassifier

###################################################################################################################################################

def get_enc_emb_str(mdl, band_names):
	dims = mdl.get_embd_dims_list()
	if type(dims)==dict:
		txts = ['-'.join([f'{b}{d}' for d in dims[b]]) for b in band_names]
		return '.'.join(txts)
	else:
		txts = [f'{d}' for d in dims]
		return '-'.join(txts)

###################################################################################################################################################

class ModelBaseline(nn.Module):
	def __init__(self, **raw_kwargs):
		super().__init__()

	def get_output_dims(self):
		return self.encoder.get_output_dims()

	def get_finetuning_parameters(self):
		finetuning_parameters = [self.classifier] # self self.classifier
		return finetuning_parameters

	def forward(self, tdict:dict, **kwargs):
		encoder_tdict = self.autoencoder['encoder'](tdict, **kwargs)
		decoder_tdict = self.autoencoder['decoder'](encoder_tdict)
		classifier_tdict = self.classifier(encoder_tdict)
		return classifier_tdict

###################################################################################################################################################

class SerialTimeModSelfAttnModel(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()
		### attributes
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.dummy_seft = self.mdl_kwargs['dummy_seft']
		self.band_names = self.mdl_kwargs['band_names']

		### encoder
		enc_embd_dims = self.mdl_kwargs['embd_dims']
		encoder = attn_encoders.TimeModSelfAttnEncoderS(**self.mdl_kwargs)
		
		### decoder
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		decoder = DECODER_CLASS(**dec_mdl_kwargs)

		### model
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':enc_embd_dims})
		self.classifier = CLASSIFIER_CLASS(**self.class_mdl_kwargs)

	def get_info(self):
		return {
			'encoder':self.autoencoder['encoder'].get_info(),
			}

	def get_name(self):
		return get_model_name({
			'mdl':f'SerialTimeModAttn',
			'input_dims':f'{self.input_dims}',
			'dummy_seft':f'{self.dummy_seft}',
			'm':self.mdl_kwargs['te_features'],
			'kernel_size':self.mdl_kwargs['kernel_size'],
			'heads':self.mdl_kwargs['heads']*len(self.band_names), # H*B
			'time_noise_window':self.mdl_kwargs['time_noise_window'],
			'enc_emb':get_enc_emb_str(self.autoencoder['encoder'], self.band_names),
			'dec_emb':get_enc_emb_str(self.autoencoder['decoder'], self.band_names),
		})

class ParallelTimeModSelfAttnModel(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### attributes
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.dummy_seft = self.mdl_kwargs['dummy_seft']
		self.band_names = self.mdl_kwargs['band_names']

		### encoder
		enc_embd_dims = self.mdl_kwargs['embd_dims']
		encoder = attn_encoders.TimeModSelfAttnEncoderP(**self.mdl_kwargs)
			
		### decoder
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		decoder = DECODER_CLASS(**dec_mdl_kwargs)
		
		### model
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':enc_embd_dims})
		self.classifier = CLASSIFIER_CLASS(**self.class_mdl_kwargs)

	def get_info(self):
		return {
			'encoder':self.autoencoder['encoder'].get_info(),
			}

	def get_name(self):
		return get_model_name({
			'mdl':f'ParallelTimeModAttn',
			'input_dims':f'{self.input_dims}',
			'dummy_seft':f'{self.dummy_seft}',
			'm':self.mdl_kwargs['te_features'],
			'kernel_size':self.mdl_kwargs['kernel_size'],
			'heads':self.mdl_kwargs['heads'], # H
			'time_noise_window':self.mdl_kwargs['time_noise_window'],
			'enc_emb':get_enc_emb_str(self.autoencoder['encoder'], self.band_names),
			'dec_emb':get_enc_emb_str(self.autoencoder['decoder'], self.band_names),
		})

###################################################################################################################################################

class SerialRNNModel(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### attributes
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.dummy_seft = self.mdl_kwargs['dummy_seft']
		self.band_names = self.mdl_kwargs['band_names']
		self.rnn_cell_name = self.mdl_kwargs['rnn_cell_name']

		### encoder
		enc_embd_dims = self.mdl_kwargs['embd_dims']
		encoder = rnn_encoders.RNNEncoderS(**self.mdl_kwargs)

		### decoder
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		decoder = DECODER_CLASS(**dec_mdl_kwargs)
		
		### model
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':enc_embd_dims})
		self.classifier = CLASSIFIER_CLASS(**self.class_mdl_kwargs)

	def get_name(self):
		return get_model_name({
			'mdl':f'SerialRNN',
			'input_dims':f'{self.input_dims}',
			'dummy_seft':f'{self.dummy_seft}',
			'enc_emb':get_enc_emb_str(self.autoencoder['encoder'], self.band_names),
			'dec_emb':get_enc_emb_str(self.autoencoder['decoder'], self.band_names),
			'cell':f'{self.rnn_cell_name}',
		})

class ParallelRNNModel(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### attributes
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.dummy_seft = self.mdl_kwargs['dummy_seft']
		self.band_names = self.mdl_kwargs['band_names']
		self.rnn_cell_name = self.mdl_kwargs['rnn_cell_name']

		### encoder
		enc_embd_dims = self.mdl_kwargs['embd_dims']
		encoder = rnn_encoders.RNNEncoderP(**self.mdl_kwargs)
		
		### decoder
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		decoder = DECODER_CLASS(**dec_mdl_kwargs)

		### model
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':enc_embd_dims})
		self.classifier = CLASSIFIER_CLASS(**self.class_mdl_kwargs)

	def get_name(self):
		return get_model_name({
			'mdl':f'ParallelRNN',
			'input_dims':f'{self.input_dims}',
			'dummy_seft':f'{self.dummy_seft}',
			'enc_emb':get_enc_emb_str(self.autoencoder['encoder'], self.band_names),
			'dec_emb':get_enc_emb_str(self.autoencoder['decoder'], self.band_names),
			'cell':f'{self.rnn_cell_name}',
		})

###################################################################################################################################################

class SerialTimeCatSelfAttnModel(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()
		### attributes
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.dummy_seft = self.mdl_kwargs['dummy_seft']
		self.band_names = self.mdl_kwargs['band_names']

		### encoder
		enc_embd_dims = self.mdl_kwargs['embd_dims']
		encoder = attn_extra_encoders.TimeCatSelfAttnEncoderS(**self.mdl_kwargs)
		
		### decoder
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		decoder = DECODER_CLASS(**dec_mdl_kwargs)

		### model
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':enc_embd_dims})
		self.classifier = CLASSIFIER_CLASS(**self.class_mdl_kwargs)

	def get_name(self):
		return get_model_name({
			'mdl':f'SerialTimeCatAttn',
			'input_dims':f'{self.input_dims}',
			'dummy_seft':f'{self.dummy_seft}',
			'm':self.mdl_kwargs['te_features'],
			'heads':self.mdl_kwargs['heads']*len(self.band_names), # H*B
			'enc_emb':get_enc_emb_str(self.autoencoder['encoder'], self.band_names),
			'dec_emb':get_enc_emb_str(self.autoencoder['decoder'], self.band_names),
		})

class ParallelTimeCatSelfAttnModel(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### attributes
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.dummy_seft = self.mdl_kwargs['dummy_seft']
		self.band_names = self.mdl_kwargs['band_names']

		### encoder
		enc_embd_dims = self.mdl_kwargs['embd_dims']
		encoder = attn_extra_encoders.TimeCatSelfAttnEncoderP(**self.mdl_kwargs)
			
		### decoder
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		decoder = DECODER_CLASS(**dec_mdl_kwargs)
		
		### model
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':enc_embd_dims})
		self.classifier = CLASSIFIER_CLASS(**self.class_mdl_kwargs)

	def get_name(self):
		return get_model_name({
			'mdl':f'ParallelTimeCatAttn',
			'input_dims':f'{self.input_dims}',
			'dummy_seft':f'{self.dummy_seft}',
			'm':self.mdl_kwargs['te_features'],
			'heads':self.mdl_kwargs['heads'], # H
			'enc_emb':get_enc_emb_str(self.autoencoder['encoder'], self.band_names),
			'dec_emb':get_enc_emb_str(self.autoencoder['decoder'], self.band_names),
		})

###################################################################################################################################################

class SerialTimeModRNNModel(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### attributes
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.dummy_seft = self.mdl_kwargs['dummy_seft']
		self.band_names = self.mdl_kwargs['band_names']
		self.rnn_cell_name = self.mdl_kwargs['rnn_cell_name']

		### encoder
		enc_embd_dims = self.mdl_kwargs['embd_dims']
		encoder = rnn_extra_encoders.TimeModRNNEncoderS(**self.mdl_kwargs)

		### decoder
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		decoder = DECODER_CLASS(**dec_mdl_kwargs)
		
		### model
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':enc_embd_dims})
		self.classifier = CLASSIFIER_CLASS(**self.class_mdl_kwargs)

	def get_name(self):
		return get_model_name({
			'mdl':f'SerialTimeModRNN',
			'input_dims':f'{self.input_dims}',
			'dummy_seft':f'{self.dummy_seft}',
			'enc_emb':get_enc_emb_str(self.autoencoder['encoder'], self.band_names),
			'dec_emb':get_enc_emb_str(self.autoencoder['decoder'], self.band_names),
			'cell':f'{self.rnn_cell_name}',
		})

class ParallelTimeModRNNModel(ModelBaseline):
	def __init__(self, **raw_kwargs):
		super().__init__()

		### attributes
		for name, val in raw_kwargs.items():
			setattr(self, name, val)

		self.input_dims = self.mdl_kwargs['input_dims']
		self.dummy_seft = self.mdl_kwargs['dummy_seft']
		self.band_names = self.mdl_kwargs['band_names']
		self.rnn_cell_name = self.mdl_kwargs['rnn_cell_name']

		### encoder
		enc_embd_dims = self.mdl_kwargs['embd_dims']
		encoder = rnn_extra_encoders.TimeModRNNEncoderP(**self.mdl_kwargs)
		
		### decoder
		dec_mdl_kwargs = deepcopy(self.mdl_kwargs)
		decoder = DECODER_CLASS(**dec_mdl_kwargs)

		### model
		self.autoencoder = nn.ModuleDict({'encoder':encoder, 'decoder':decoder})
		self.class_mdl_kwargs.update({'input_dims':enc_embd_dims})
		self.classifier = CLASSIFIER_CLASS(**self.class_mdl_kwargs)

	def get_name(self):
		return get_model_name({
			'mdl':f'ParallelTimeModRNN',
			'input_dims':f'{self.input_dims}',
			'dummy_seft':f'{self.dummy_seft}',
			'enc_emb':get_enc_emb_str(self.autoencoder['encoder'], self.band_names),
			'dec_emb':get_enc_emb_str(self.autoencoder['decoder'], self.band_names),
			'cell':f'{self.rnn_cell_name}',
		})