from __future__ import print_function
from __future__ import division
from . import _C

import math
import torch
import torch.tensor as Tensor
import numpy as np
from torch.utils.data import Dataset
import random
from .scalers import CustomStandardScaler, LogStandardScaler
import fuzzytools.strings as strings
from fuzzytools.lists import get_random_item
import fuzzytools.files as ftfiles
from fuzzytools.progress_bars import ProgressBar
from fuzzytorch.utils import TDictHolder, print_tdict
import fuzzytorch.models.seq_utils as seq_utils
from lchandler.lc_classes import diff_vector
from nested_dict import nested_dict
from copy import copy, deepcopy

###################################################################################################################################################

def save(obj, filedir):
	assert isinstance(obj, dict)
	ftfiles.create_dir('/'.join(filedir.split('/')[:-1]))
	# torch.save(items, items_filedir)
	# ftfiles.save_pickle(filedir, obj)
	np.save(filedir, obj)

def load(filedir):
	# torch.load(self.precomputed_filedirs_dict[lcobj_name])
	# return ftfiles.load_pickle(filedir)
	return np.load(f'{filedir}.npy', allow_pickle=True).item()

def numpy_to_tensor(tdict):
	for key in tdict.keys():
		tdict[key] = torch.from_numpy(tdict[key])
	return tdict

def _get_numpy_item(args):
	tdict, lcobj = CustomDataset.get_numpy_item(*args)
	return tdict

def _get_item(args):
	tdict, lcobj = CustomDataset.get_item(*args)
	return tdict

###################################################################################################################################################

class CustomDataset(Dataset):
	def __init__(self, lcset_name, lcset,
		precomputed_mode='device',
		device='cpu',
		in_attrs=None,
		rec_attr=None,
		max_day:float=np.infty,
		precomputed_copies=1,
		uses_daugm=False,
		uses_dynamic_balance=False,
		batch_prop=1,
		da_kwargs={},
		):
		assert precomputed_copies>=1
		assert precomputed_mode in ['online', 'disk', 'device']
		assert device in ['cpu', 'cuda:0']

		self.lcset_name = lcset_name
		self.lcset = lcset
		self.precomputed_mode = precomputed_mode
		self.device = device
		self.in_attrs = in_attrs
		self.rec_attr = rec_attr
		self.max_day = max_day
		self.precomputed_copies = precomputed_copies
		self.uses_daugm = uses_daugm
		self.uses_dynamic_balance = uses_dynamic_balance
		self.batch_prop = batch_prop
		self.da_kwargs = da_kwargs
		self.reset()

	def reset(self):
		torch.cuda.empty_cache()
		self.lcset_info = self.lcset.get_info()
		self.band_names = self.lcset.band_names
		self.class_names = self.lcset.class_names
		self.survey = self.lcset.survey
		self._max_day = self.max_day # save original to perform reset

		self.prepare_data_for_statistics()
		self.max_len = self.calcule_max_len()

		### scalers computations
		self.scalers = nested_dict()
		self.calcule_dtime_scaler()
		self.calcule_in_scaler()
		self.calcule_rec_scaler()
		print(self.lcset_name, self.scalers['dtime']['g'].m, self.scalers['dtime']['g'].s)
		self.scalers = self.scalers.to_dict()

		self.calcule_poblation_weights()
		self.calcule_balanced_w_cdict()

		self.lcset.generate_boostrap(batch_prop=self.batch_prop)
		self.pre_epoch_step()

	def calcule_precomputed(self,
		verbose=0,
		disk_location='../temp/datasets',
		read_from_disk=False,
		):
		lcobj_names = self.lcset.get_lcobj_names()
		ds_prob = self.da_kwargs.get('ds_prob', None)
		bypass_prob = self.da_kwargs.get('bypass_prob', None)
		disk_rootdir = f'{disk_location}/{self.lcset_name}/bypass_prob={bypass_prob}~ds_prob={ds_prob}'

		if self.precomputed_mode=='online':
			return

		if self.precomputed_mode=='disk':
			self.precomputed_filedirs_dict = {}
			bar = ProgressBar(len(lcobj_names), dummy=verbose==0)
			for lcobj_name in lcobj_names:
				bar(f'lcset_name={self.lcset_name}; precomputed_copies={self.precomputed_copies}; disk_rootdir={disk_rootdir}; lcobj_name={lcobj_name}')
				items_filedirs = [f'{disk_rootdir}/{lcobj_name}.{k}' for k in range(0, self.precomputed_copies)]
				self.precomputed_filedirs_dict[lcobj_name] = items_filedirs
				for items_filedir in items_filedirs:
					if read_from_disk or ftfiles.filedir_exists(items_filedir):
						pass
					else:
						in_tdict = _get_numpy_item((self, lcobj_name))
						save(in_tdict, items_filedir)
			bar.done()
			return

		if self.precomputed_mode=='device':
			self.precomputed_dict = {lcobj_name:[] for lcobj_name in lcobj_names}
			bar = ProgressBar(len(lcobj_names), dummy=verbose==0)
			for lcobj_name in lcobj_names:
				bar(f'lcset_name={self.lcset_name}; device={self.device}; lcobj_name={lcobj_name}')
				for k in range(0, self.precomputed_copies):
					items_filedir = f'{disk_rootdir}/{lcobj_name}.{k}'
					in_tdict = numpy_to_tensor(load(items_filedir)) if read_from_disk else _get_item((self, lcobj_name))
					self.precomputed_dict[lcobj_name] += [TDictHolder(in_tdict).to(self.device)]
			bar.done()
			return

	def prepare_data_for_statistics(self):
		### extra merged band * used to statistics as scales for serial
		self.lcset.reset_all_day_offset_serial()
		self.lcset.generate_all_mb()
		
	def calcule_max_len(self):
		lens = []
		for lcobj_name in self.get_lcobj_names():
			lcobjb = copy(self.lcset[lcobj_name].get_b('*')) # copy
			lcobjb.clip_attrs_given_max_day(self.max_day)
			lens += [len(lcobjb)]
		return max(lens)

###################################################################################################################################################

	def reset_max_day(self):
		self.max_day = self._max_day

	def calcule_balanced_w_cdict(self):
		self.balanced_w_cdict = self.lcset.get_class_balanced_weights_cdict()

	def calcule_poblation_weights(self):
		self.nof_samples_cdict = self.lcset.get_nof_samples_cdict()

	def get_output_dims(self):
		return len(self.in_attrs)

	def extra_repr(self):
		return strings.get_string_from_dict({
			'lcset_len':f'{len(self.lcset):,}',
			'class_names':self.class_names,
			'band_names':self.band_names,
			'max_day':f'{self.max_day:.2f}',
			'max_len': f'{self.max_len:,}',
			'in_attrs':self.in_attrs,
			'rec_attr':self.rec_attr,
			'balanced_w_cdict':self.balanced_w_cdict,
			'nof_samples_cdict':self.nof_samples_cdict,
			}, ', ', '=')

	def __repr__(self):
		txt = f'CustomDataset({self.extra_repr()})'
		return txt

	def get_max_len(self):
		return self.max_len

	def set_max_day(self, max_day):
		assert max_day<=self._max_day
		self.max_day = max_day

	def set_scalers_from(self, other):
		self.set_scalers(other.get_scalers())
		return self

	def get_scalers(self):
		return self.scalers

	def set_scalers(self, scalers):
		self.scalers = {}
		for k in scalers.keys():
			self.scalers[k] = deepcopy(scalers[k])

###################################################################################################################################################

	def calcule_dtime_scaler(self):
		for kb,b in enumerate(self.band_names+['*']):
			values = self.lcset.get_all_parallel_diff_days_b(b, generates_mb=False)[...,None] # (n,1)
			scaler = CustomStandardScaler()
			# scaler = LogStandardScaler()
			# scaler = LogQuantileTransformer() # slow
			scaler.fit(values)
			self.scalers['dtime'][b] = scaler

	def calcule_in_scaler(self):
		for kb,b in enumerate(self.band_names+['*']):
			values = np.concatenate([self.lcset.get_all_values_b(b, in_attr)[...,None] for in_attr in self.in_attrs], axis=-1) # (n,f)
			# scaler = CustomStandardScaler()
			scaler = LogStandardScaler()
			# scaler = LogQuantileTransformer() # slow
			scaler.fit(values)
			self.scalers['in'][b] = scaler

	def calcule_rec_scaler(self):
		for kb,b in enumerate(self.band_names+['*']):
			values = self.lcset.get_all_values_b(b, self.rec_attr)[...,None] # (n,1)
			# scaler = CustomStandardScaler()
			scaler = LogStandardScaler()
			# scaler = LogQuantileTransformer() # slow
			scaler.fit(values)
			self.scalers['rec'][b] = scaler

###################################################################################################################################################

	def dtime_normalize(self, x, b):
		'''
		x (t,1)
		'''
		if len(x)==0:
			return x
		assert len(x.shape)==2
		assert x.shape[-1]==1
		return self.scalers['dtime'][b].transform(x) # (t,1)

	def in_normalize(self, x, b):
		'''
		x (t,f)
		'''
		if len(x)==0:
			return x
		assert len(x.shape)==2
		assert x.shape[-1]==len(self.in_attrs)
		return self.scalers['in'][b].transform(x) # (t,f)
	
	def rec_normalize(self, x, b):
		'''
		x (t,1)
		'''
		if len(x)==0:
			return x
		assert len(x.shape)==2
		assert x.shape[-1]==1
		return self.scalers['rec'][b].transform(x) # (t,1)

	def get_rec_inverse_transform(self, model_rec_x_b, b):
		'''
		x (t)
		'''
		assert len(model_rec_x_b.shape)==1
		return self.scalers['rec'][b].inverse_transform(model_rec_x_b[...,None])[...,0]

	###################################################################################################################################################

	def get_random_stratified_lcobj_names(self,
		nc=1,
		):
		# stratified, mostly used for images in experiments
		lcobj_names = []
		random_ndict = self.lcset.get_random_stratified_lcobj_names(nc)
		for c in self.class_names:
			lcobj_names += random_ndict[c]
		return lcobj_names

	###################################################################################################################################################

	def get_lcobj_names(self):
		self.lcset_lcobj_names = self.lcset.get_lcobj_names()
		return self.lcset_lcobj_names

	def get_train_lcobj_names(self):
		if self.uses_dynamic_balance:
			train_lcobj_names = self.balanced_lcobj_names
		else:
			train_lcobj_names = self.get_lcobj_names()
		# print([train_lcobj_names[int(idx)] for idx in np.linspace(0, len(train_lcobj_names)-1, 12)])
		return train_lcobj_names

	def pre_epoch_step(self):
		if self.uses_dynamic_balance:
			self.balanced_lcobj_names = self.lcset.get_boostrap_samples() # important

	def __len__(self):
		return len(self.get_train_lcobj_names())

	###################################################################################################################################################

	def fix_tdict(self, tdict):
		for key in tdict.keys():
			if list(tdict[key].shape)==[1]: # (1)
				tdict[key] = tdict[key][0]
			if len(tdict[key].shape)==2: # (t,f)
				tdict[key] = seq_utils.get_seq_clipped_shape(tdict[key], self.max_len)
		return tdict

	def __getitem__(self, idx:int):
		# if idx==0:
			# print(self.lcset_name, self.scalers['dtime']['g'].m, self.scalers['dtime']['g'].s)
		lcobj_name = self.get_train_lcobj_names()[idx]
		if self.precomputed_mode=='online':
			tdict, _ = self.get_item(lcobj_name) # cpu

		elif self.precomputed_mode=='disk':
			items_filedirs = self.precomputed_filedirs_dict[lcobj_name]
			items_filedir = get_random_item(items_filedirs)
			tdict = numpy_to_tensor(load(items_filedir))

		elif self.precomputed_mode=='device':
			items = self.precomputed_dict[lcobj_name]
			tdict = get_random_item(items)

		tdict = self.fix_tdict(tdict)
		return tdict

	def get_item(self, lcobj_name):
		tdict, lcobj = self.get_numpy_item(lcobj_name)
		return numpy_to_tensor(tdict), lcobj

	def get_numpy_item(self, lcobj_name):
		'''
		apply data augmentation, this overrides obj information
		be sure to copy the input lcobj!!!!
		'''
		lcobj = copy(self.lcset[lcobj_name]) # copy
		### perform da ignoring merged band *
		if self.uses_daugm:
			for kb,b in enumerate(self.band_names):
				lcobjb = lcobj.get_b(b)
				lcobjb.apply_data_augmentation( # curve points downsampling we need to ensure the model to see compelte curves
					**self.da_kwargs,
					)

		### clip by max day
		lcobj.reset_day_offset_serial() # remove day offset!
		lcobj.clip_attrs_given_max_day(self.max_day) # clip by day
		parallel_diff_days = lcobj.get_parallel_diff_days(generates_mb=True) # important to re-generate merged band *

		###
		s_onehot = lcobj.get_onehot_serial() # (t,b)
		# print(s_onehot.shape, s_onehot)
		tdict = {}
		tdict[f'input/s_onehot'] = s_onehot # (t,b)
		for kb,b in enumerate(self.band_names+['*']):
			lcobjb = lcobj.get_b(b)

			onehot = np.ones((len(lcobjb),), dtype=bool)[...,None] # (t,1)
			rx = lcobjb.get_custom_x(self.in_attrs) # raw_x (t,f)
			x = self.in_normalize(rx, b) # norm_x (t,f)
			rtime = lcobjb.days[...,None] # raw_time (t,1)
			rtime_pb = copy(rtime)-rtime[0] if len(rtime)>0 else copy(rtime) # raw_time_pb (t,1)
			# time
			rdtime = parallel_diff_days[b][...,None] # raw_dtime (t,1)
			dtime = self.dtime_normalize(rdtime, b) # norm_dtime (t,1)
			# for preserved band 
			rdtime_pb = copy(rdtime) # raw_dtime_pb (t,1)
			if len(rdtime_pb)>0:
				rdtime_pb[0] = 0
			dtime_pb = self.dtime_normalize(rdtime_pb, b) # norm_dtime_pb (t,1)

			# print(b, f'rtime={rtime}')
			# print(b, f'rtime_pb={rtime_pb}')

			# print(b, f'rdtime={rdtime}')
			# print(b, f'dtime={dtime}')
			# print(b, f'rdtime_pb={rdtime_pb}')
			# print(b, f'dtime_pb={dtime_pb}')

			tdict[f'input/onehot.{b}'] = onehot
			tdict[f'input/x.{b}'] = x
			#time
			tdict[f'input/rtime.{b}'] = rtime
			tdict[f'input/rtime_pb.{b}'] = rtime_pb
			# tdict[f'input/rdtime.{b}'] = rdtime
			tdict[f'input/dtime.{b}'] = dtime
			# tdict[f'input/rdtime_pb.{b}'] = rdtime_pb
			tdict[f'input/dtime_pb.{b}'] = dtime_pb

			rrecx = lcobjb.get_custom_x([self.rec_attr]) # raw_recx (t,1)
			recx = self.rec_normalize(rrecx, b) # norm_recx (t,1)

			rerror = lcobjb.obse[...,None] # raw_error (t,1)
			assert np.all(rerror>=0)

			tdict[f'target/recx.{b}'] = recx
			tdict[f'target/rerror.{b}'] = rerror

		tdict[f'target/y'] = np.array([lcobj.y], dtype=np.int64) # (1)
		tdict[f'target/balanced_w'] = np.array([self.balanced_w_cdict[self.class_names[lcobj.y]]], dtype=np.float32) # # 1/(N_c*C) (1)
		# assert 0
		return tdict, lcobj