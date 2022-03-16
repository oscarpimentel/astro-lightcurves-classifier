#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-torch') # or just install the module
sys.path.append('../../fuzzy-tools') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module

###################################################################################################################################################
import argparse
from fuzzytools.prints import print_big_bar

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method',  type=str, default='spm-mcmc-estw')
parser.add_argument('--gpu',  type=int, default=-1)
parser.add_argument('--mc',  type=str, default='parallel_rnn_models')
parser.add_argument('--batch_size',  type=int, default=32)
parser.add_argument('--save_rootdir',  type=str, default='../save')
parser.add_argument('--mid',  type=str, default='0')
parser.add_argument('--kf',  type=str, default='0')
parser.add_argument('--bypass_synth',  type=int, default=0) # 0 1
parser.add_argument('--bypass_pre_training',  type=int, default=0) # 0 1
parser.add_argument('--invert_mpg',  type=int, default=0) # 0 1
parser.add_argument('--only_perform_exps',  type=int, default=0) # 0 1
parser.add_argument('--perform_slow_exps',  type=int, default=0) # 0 1
parser.add_argument('--extra_model_name',  type=str, default='')
parser.add_argument('--num_workers',  type=int, default=12)
parser.add_argument('--pin_memory',  type=int, default=1) # 0 1
parser.add_argument('--classifier_mids',  type=int, default=2)
parser.add_argument('--pt_balanced_metrics',  type=int, default=1)
parser.add_argument('--ft_balanced_metrics',  type=int, default=1)
parser.add_argument('--precompute_only',  type=int, default=0)
parser.add_argument('--dummy_seft',  type=int, default=1)
parser.add_argument('--preserved_band',  type=str, default='.')
parser.add_argument('--bypass_prob',  type=float, default=.0)
parser.add_argument('--ds_prob',  type=float, default=.1)
#main_args = parser.parse_args([])
main_args = parser.parse_args()
print_big_bar()

###################################################################################################################################################
from fuzzytools.files import search_for_filedirs
from lchandler import _C as _C

surveys_rootdir = '../../surveys-save/'
filedirs = search_for_filedirs(surveys_rootdir, fext=_C.EXT_SPLIT_LIGHTCURVE)

###################################################################################################################################################
import numpy as np
from fuzzytools.files import load_pickle, save_pickle
from fuzzytools.files import get_dict_from_filedir

filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={main_args.method}.splcds'
filedict = get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
lcdataset = load_pickle(filedir)
lcdataset.only_keep_kf(main_args.kf) # saves ram
#print(lcdataset)

###################################################################################################################################################
from lcclassifier.models.model_collections import ModelCollections

model_collections = ModelCollections(lcdataset)
getattr(model_collections, main_args.mc)()

#getattr(model_collections, 'parallel_rnn_models_dt')()
#getattr(model_collections, 'parallel_rnn_models_te')() # not used
#getattr(model_collections, 'serial_rnn_models_dt')()
#getattr(model_collections, 'serial_rnn_models_te')() # not used

#getattr(model_collections, 'parallel_tcnn_models_dt')()
#getattr(model_collections, 'parallel_tcnn_models_te')() # not used
#getattr(model_collections, 'serial_tcnn_models_dt')()
#getattr(model_collections, 'serial_tcnn_models_te')() # not used

#getattr(model_collections, 'parallel_atcnn_models_te')()
#getattr(model_collections, 'serial_atcnn_models_te')()

print(model_collections)

###################################################################################################################################################
### LOSS & METRICS
import lcclassifier.losses as losses
import lcclassifier.metrics as metrics
from fuzzytorch.metrics import LossWrapper

loss_kwargs = {
	'class_names':lcdataset[lcdataset.get_lcset_names()[0]].class_names,
	'band_names':lcdataset[lcdataset.get_lcset_names()[0]].band_names,
	'target_is_onehot':False,
	'preserved_band':main_args.preserved_band,
	}
	
### pt
pt_loss = losses.LCCompleteLoss('wmse+binxentropy', None, **loss_kwargs)
weight_key = f'target/balanced_w' if main_args.pt_balanced_metrics else None
pt_metrics = [
	LossWrapper(losses.LCCompleteLoss(('b-' if main_args.pt_balanced_metrics else '')+'wmse+binxentropy', weight_key, **loss_kwargs)),
	LossWrapper(losses.LCBinXEntropy(('b-' if main_args.pt_balanced_metrics else '')+'binxentropy', weight_key, **loss_kwargs)),
	metrics.LCAccuracy(('b-' if main_args.pt_balanced_metrics else '')+'accuracy', weight_key, **loss_kwargs),
	]

### ft
ft_loss = losses.LCBinXEntropy('binxentropy', None, **loss_kwargs)
weight_key = f'target/balanced_w' if main_args.ft_balanced_metrics else None
ft_metrics = [
	LossWrapper(losses.LCBinXEntropy(('b-' if main_args.ft_balanced_metrics else '')+'binxentropy', weight_key, **loss_kwargs)),
	metrics.LCAccuracy(('b-' if main_args.ft_balanced_metrics else '')+'accuracy', weight_key, **loss_kwargs),
	]

###################################################################################################################################################
import os

if main_args.gpu>=0:
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # see issue #152
	os.environ['CUDA_VISIBLE_DEVICES'] = str(main_args.gpu) # CUDA-GPU
	device = 'cuda:0'
else:
	device = 'cpu'

###################################################################################################################################################
mp_grids = model_collections.mps[::-1] if main_args.invert_mpg else model_collections.mps
for mp_grid in mp_grids: # MODEL CONFIGS
	from lcclassifier.datasets import CustomDataset
	from torch.utils.data import DataLoader
	from fuzzytools.strings import get_dict_from_string
	from fuzzytools.files import get_filedirs, copy_filedir
	import torch
	from copy import copy, deepcopy

	### DATASETS
	dataset_kwargs = mp_grid['dataset_kwargs']
	s_train_dataset_opt_kwargs = {
		'uses_dynamic_balance':True,
		'batch_prop':.5,
		'precomputed_copies':3,
		'uses_daugm':True,
		'da_kwargs':{
			'ds_prob':main_args.ds_prob,
			'std_scale':1/2,
			'bypass_prob':main_args.bypass_prob,
			}
		}
	s_train_dataset_opt_kwargs.update(dataset_kwargs)

	lcset_name = f'{main_args.kf}@train.{main_args.method}' if not main_args.bypass_synth else f'{main_args.kf}@train'
	s_train_dataset_opt = CustomDataset(lcset_name, lcdataset[lcset_name],
		precomputed_mode='disk', # disk online device
		# precomputed_mode='online', # disk online device
		device='cpu',
		**s_train_dataset_opt_kwargs)

	s_train_loader_opt = DataLoader(s_train_dataset_opt,
		shuffle=True,
		batch_size=main_args.batch_size,
		drop_last=True,
		num_workers=main_args.num_workers,
		pin_memory=main_args.pin_memory,
		worker_init_fn=lambda id:np.random.seed(torch.initial_seed()//2**32+id), # num_workers-numpy bug
		# persistent_workers=main_args.num_workers>0, # HUGE BUG WHEN USED
		)

	lcset_name = f'{main_args.kf}@train'
	r_train_dataset = CustomDataset(lcset_name, lcdataset[lcset_name],
		precomputed_mode='device',
		device=device,
		**dataset_kwargs)
	r_train_loader = DataLoader(r_train_dataset, shuffle=False, batch_size=main_args.batch_size)

	lcset_name = f'{main_args.kf}@val'
	r_val_dataset = CustomDataset(lcset_name, lcdataset[lcset_name],
		precomputed_mode='device',
		device=device,
		**dataset_kwargs)
	r_val_loader = DataLoader(r_val_dataset, shuffle=False, batch_size=main_args.batch_size)

	lcset_name = f'{main_args.kf}@test'
	r_test_dataset = CustomDataset(lcset_name, lcdataset[lcset_name],
		precomputed_mode='device',
		device=device,
		**dataset_kwargs)
	r_test_loader = DataLoader(r_test_dataset, shuffle=False, batch_size=main_args.batch_size)

	### compute datasets
	s_train_dataset_opt.set_scalers_from(s_train_dataset_opt).calcule_precomputed(verbose=1, read_from_disk=not main_args.precompute_only); print(s_train_dataset_opt)
	if main_args.precompute_only:
		assert 0 # exit
	r_train_dataset.set_scalers_from(s_train_dataset_opt).calcule_precomputed(verbose=1); print(r_train_dataset)
	r_val_dataset.set_scalers_from(s_train_dataset_opt).calcule_precomputed(verbose=1); print(r_val_dataset)
	r_test_dataset.set_scalers_from(s_train_dataset_opt).calcule_precomputed(verbose=1); print(r_test_dataset)

	### GET MODEL
	mp_grid['mdl_kwargs']['input_dims'] = r_test_loader.dataset.get_output_dims()
	mp_grid['mdl_kwargs']['dummy_seft'] = main_args.dummy_seft
	mp_grid['mdl_kwargs']['preserved_band'] = main_args.preserved_band
	model = mp_grid['mdl_kwargs']['C'](**mp_grid) # model creation

	### pre-training
	### OPTIMIZER
	import torch.optim as optims
	from fuzzytorch.optimizers import LossOptimizer
	import math

	def pt_lr_f(epoch):
		min_lr, max_lr = 1e-10, 1.1e-3
		d_epochs = 10
		exp_decay_k = 0
		p = np.clip(epoch/d_epochs, 0, 1) # 0 > 1
		lr = (1-p)*min_lr+p*max_lr
		lr = math.exp(-np.clip(epoch-d_epochs, 0, None)*exp_decay_k)*lr
		return lr

	pt_opt_kwargs_f = {
		'lr':pt_lr_f,
		'weight_decay':lambda epoch:.2e-3,
		}
	pt_optimizer = LossOptimizer(model, optims.Adam, pt_opt_kwargs_f, # SGD Adagrad Adadelta RMSprop Adam AdamW
		clip_grad=1,
		)

	### MONITORS
	from fuzzytools.prints import print_bar
	from fuzzytorch.handlers import ModelTrainHandler
	from fuzzytorch.monitors import LossMonitor
	from fuzzytorch import _C as _Cfuzzytorch
	from lcclassifier import _C as _Clcclassifier
	import math

	pt_loss_monitors = LossMonitor(pt_loss, pt_optimizer, pt_metrics,
		val_epoch_counter_duration=0, # every k epochs check
		earlystop_epoch_duration=1e6,
		target_metric_crit=('b-' if main_args.pt_balanced_metrics else '')+'wmse+binxentropy',
		#save_mode=_Cfuzzytorch.SM_NO_SAVE,
		#save_mode=_Cfuzzytorch.SM_ALL,
		#save_mode=_Cfuzzytorch.SM_ONLY_ALL,
		save_mode=_Cfuzzytorch.SM_ONLY_INF_METRIC,
		#save_mode=_Cfuzzytorch.SM_ONLY_INF_LOSS,
		#save_mode=_Cfuzzytorch.SM_ONLY_SUP_METRIC,
		)

	### TRAIN
	train_mode = 'pre-training'
	extra_model_name_dict = {
			'b':f'{main_args.batch_size}',
			'pb':f'{main_args.preserved_band}',
			'bypass_synth':f'{main_args.bypass_synth}',
			'bypass_prob':f'{main_args.bypass_prob}',
			'ds_prob':f'{main_args.ds_prob}',
			}
	extra_model_name_dict.update(get_dict_from_string(main_args.extra_model_name))
	pt_model_train_handler = ModelTrainHandler(model, pt_loss_monitors,
		id=main_args.mid,
		epochs_max=160, # limit this as the pre-training is very time consuming
		extra_model_name_dict=extra_model_name_dict,
		)
	complete_model_name = pt_model_train_handler.get_complete_model_name()
	pt_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
	pt_model_train_handler.build_gpu(device)
	print(pt_model_train_handler)
	if main_args.only_perform_exps or main_args.bypass_pre_training:
		pass
	else:
		pt_model_train_handler.fit_loader(s_train_loader_opt, {
			# 'train':r_train_loader,
			'val':r_val_loader,
			},
			train_dataset_method_call='pre_epoch_step',
			) # main fit
	del s_train_loader_opt # to save memory
	pt_model_train_handler.load_model() # important, refresh to best model

	###################################################################################################################################################
	import fuzzytorch
	import fuzzytorch.plots
	import fuzzytorch.plots.training as ffplots

	### training plots
	plot_kwargs = {
		'save_rootdir':f'../save/training_plots',
		}
	#ffplots.plot_loss(pt_model_train_handler, **plot_kwargs) # use this
	#ffplots.plot_evaluation_loss(train_handler, **plot_kwargs)
	#ffplots.plot_evaluation_metrics(train_handler, **plot_kwargs)

	###################################################################################################################################################
	from lcclassifier.experiments.reconstructions import save_reconstructions
	from lcclassifier.experiments.dim_reductions import save_dim_reductions
	from lcclassifier.experiments.model_info import save_model_info
	from lcclassifier.experiments.temporal_modulation import save_temporal_modulation
	from lcclassifier.experiments.performance import save_performance
	from lcclassifier.experiments.attnscores import save_attnscores_animation
	from lcclassifier.experiments.attnstats import save_attnstats
	
	pt_exp_kwargs = {
		'm':20,
		'target_is_onehot':False,
		}
	save_model_info(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/model_info/{cfilename}', **pt_exp_kwargs)

	# save_reconstructions(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs) # sanity check / slow
	# save_reconstructions(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs) # sanity check
	# save_reconstructions(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs)
	save_reconstructions(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/reconstruction/{cfilename}', **pt_exp_kwargs)

	if main_args.perform_slow_exps: # slow optional experiments
		# save_dim_reductions(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/dim_reductions/{cfilename}', model_enc_key='encz_last', **pt_exp_kwargs) # very slow
		# save_dim_reductions(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/dim_reductions/{cfilename}', model_enc_key='encz_pre_last', **pt_exp_kwargs) # very slow

		# save_attnscores_animation(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/attnscores/{cfilename}', **pt_exp_kwargs) # sanity check / slow
		# save_attnscores_animation(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/attnscores/{cfilename}', **pt_exp_kwargs) # sanity check
		# save_attnscores_animation(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/attnscores/{cfilename}', **pt_exp_kwargs)
		# save_attnscores_animation(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/attnscores/{cfilename}', **pt_exp_kwargs)

		# save_attnstats(pt_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/attnstats/{cfilename}', **pt_exp_kwargs)
		# save_attnstats(pt_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/attnstats/{cfilename}', **pt_exp_kwargs)
		# save_attnstats(pt_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/attnstats/{cfilename}', **pt_exp_kwargs)
		save_attnstats(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/attnstats/{cfilename}', **pt_exp_kwargs)

	### extra experiments
	save_temporal_modulation(pt_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/temporal_encoding/{cfilename}', **pt_exp_kwargs)
	
	###################################################################################################################################################
	###################################################################################################################################################
	###################################################################################################################################################

	### fine-tuning
	for classifier_mid in range(0, main_args.classifier_mids):
		pt_model_cache = deepcopy(pt_model_train_handler.load_model()) # copy

		lcset_name = f'{main_args.kf}@train'
		r_train_dataset_opt = CustomDataset(lcset_name, lcdataset[lcset_name],
			precomputed_mode='online',
			device='cpu',
			uses_dynamic_balance=True,
			uses_daugm=True,
			da_kwargs={
				'ds_prob':main_args.ds_prob,
				'std_scale':1/2,
				'bypass_prob':main_args.bypass_prob,
				},
			**dataset_kwargs,
			)
		r_train_loader_opt = DataLoader(r_train_dataset_opt,
			shuffle=True,
			batch_size=50,
			drop_last=True,
			num_workers=main_args.num_workers,
			pin_memory=main_args.pin_memory,
			worker_init_fn=lambda id:np.random.seed(torch.initial_seed()//2**32+id), # num_workers-numpy bug
			# persistent_workers=main_args.num_workers>0, HUGE BUG WHEN USED
			)
		r_train_dataset_opt.set_scalers_from(s_train_dataset_opt).calcule_precomputed(verbose=1)

		import torch.optim as optims
		from fuzzytorch.optimizers import LossOptimizer

		def ft_lr_f(epoch):
			min_lr, max_lr = 1e-10, 1e-3
			d_epochs = 5
			exp_decay_k = 0
			p = np.clip(epoch/d_epochs, 0, 1) # 0 > 1
			lr = (1-p)*min_lr+p*max_lr
			lr = math.exp(-np.clip(epoch-d_epochs, 0, None)*exp_decay_k)*lr
			# return max_lr
			return lr

		ft_opt_kwargs_f = {
			'lr':ft_lr_f,
			'momentum':lambda epoch:0.9,
			}
		ft_optimizer = LossOptimizer(pt_model_cache.get_finetuning_parameters(), optims.SGD, ft_opt_kwargs_f, # SGD Adagrad Adadelta RMSprop Adam AdamW
			clip_grad=1,
			)

		### monitors
		from fuzzytools.prints import print_bar
		from fuzzytorch.handlers import ModelTrainHandler
		from fuzzytorch.monitors import LossMonitor
		from fuzzytorch import _C as _Cfuzzytorch
		from lcclassifier import _C as _Clcclassifier
		import math

		ft_loss_monitors = LossMonitor(ft_loss, ft_optimizer, ft_metrics,
			val_epoch_counter_duration=0, # every k epochs check
			earlystop_epoch_duration=1e6,
			target_metric_crit=('b-' if main_args.ft_balanced_metrics else '')+'binxentropy',
			#save_mode=_Cfuzzytorch.SM_NO_SAVE,
			#save_mode=_Cfuzzytorch.SM_ALL,
			#save_mode=_Cfuzzytorch.SM_ONLY_ALL,
			save_mode=_Cfuzzytorch.SM_ONLY_INF_METRIC,
			#save_mode=_Cfuzzytorch.SM_ONLY_INF_LOSS,
			#save_mode=_Cfuzzytorch.SM_ONLY_SUP_METRIC,
			)
		### TRAIN
		train_mode = 'fine-tuning'
		extra_model_name_dict = {
				'b':f'{main_args.batch_size}',
				'pb':f'{main_args.preserved_band}',
				'bypass_synth':f'{main_args.bypass_synth}',
				'bypass_prob':f'{main_args.bypass_prob}',
				'ds_prob':f'{main_args.ds_prob}',
				}
		extra_model_name_dict.update(get_dict_from_string(main_args.extra_model_name))
		mtrain_config = {
			'id':f'{main_args.mid}c{classifier_mid}',
			'epochs_max':65,
			'save_rootdir':f'../save/{train_mode}/_training/{cfilename}',
			'extra_model_name_dict':extra_model_name_dict,
			}
		ft_model_train_handler = ModelTrainHandler(pt_model_cache, ft_loss_monitors, **mtrain_config)
		complete_model_name = ft_model_train_handler.get_complete_model_name()
		ft_model_train_handler.set_complete_save_roodir(f'../save/{complete_model_name}/{train_mode}/_training/{cfilename}/{main_args.kf}@train')
		ft_model_train_handler.build_gpu(device)
		print(ft_model_train_handler)
		if main_args.only_perform_exps:
			pass
		else:
			ft_model_train_handler.fit_loader(r_train_loader_opt, {
				# 'train':r_train_loader,
				'val':r_val_loader,
				},
				train_dataset_method_call='pre_epoch_step',
				) # main fit
		del r_train_loader_opt
		ft_model_train_handler.load_model() # important, refresh to best model

		###################################################################################################################################################
		from lcclassifier.experiments.performance import save_performance

		ft_exp_kwargs = {
			'm':15,
			'target_is_onehot':False,
			}	
		# save_performance(ft_model_train_handler, s_train_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs) # sanity check / slow
		# save_performance(ft_model_train_handler, r_train_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs) # sanity check
		# save_performance(ft_model_train_handler, r_val_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs)
		save_performance(ft_model_train_handler, r_test_loader, f'../save/{complete_model_name}/{train_mode}/performance/{cfilename}', **ft_exp_kwargs)
