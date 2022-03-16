from __future__ import print_function
from __future__ import division
from . import _C

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy, minibatch_dict_collate
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts
import fuzzytorch.models.seq_utils as seq_utils
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA, FastICA
from umap import UMAP
from umap.parametric_umap import ParametricUMAP
from fuzzytools.datascience.dim_reductors import DimReductorPipeline
import tensorflow as tf

DEFAULT_DAYS_N = _C.DEFAULT_DAYS_N
DEFAULT_MIN_DAY = _C.DEFAULT_MIN_DAY
RANDOM_STATE = None

###################################################################################################################################################

def save_dim_reductions(train_handler, data_loader, save_rootdir,
	target_is_onehot:bool=False,
	target_y_key='target/y',
	pred_y_key='model/y',
	model_enc_key='encz_last',
	days_n:int=DEFAULT_DAYS_N,
	random_state=RANDOM_STATE,
	supervised=False,
	**kwargs):
	'''
	encz_last: encoder representation-vector
	encz_pre_last: classificator layer before softmax
	'''
	train_handler.load_model() # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # always reset max day

	days_embeddings = {}
	days_y_true = {}

	thdays = np.linspace(DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
	bar = ProgressBar(len(thdays))
	with torch.no_grad():
		for thday in thdays:
			dataset.set_max_day(thday) # very important!!
			dataset.calcule_precomputed() # very important!!
			
			tdicts = []
			for ki,in_tdict in enumerate(data_loader):
				_tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device))
				tdicts += [_tdict]
			tdict = minibatch_dict_collate(tdicts)

			### class prediction
			y_true = tdict[target_y_key] # (n)
			y_pred_p = torch.nn.functional.softmax(tdict[pred_y_key], dim=-1) # (n,c)
			if target_is_onehot:
				assert y_pred_.shape==y_true.shape
				y_true = torch.argmax(y_true, dim=-1)

			# metrics
			y_true = tensor_to_numpy(y_true)
			y_pred_p = tensor_to_numpy(y_pred_p)
			days_y_true[thday] = y_true

			### embeddings
			days_embeddings[thday] = tensor_to_numpy(tdict[f'model/{model_enc_key}'])
			bar(f'thday={thday:.0f}/{thdays[-1]:.0f}; days_embeddings={days_embeddings[thday][:5,0]}')

	bar.done()

	dim_reductor = DimReductorPipeline([
		StandardScaler(),
		PCA(n_components=10),
		MinMaxScaler(feature_range=[0,1], clip=True),
		ParametricUMAP(
			n_components=2,
			metric='euclidean', # euclidean manhattan
			n_neighbors=50,
			min_dist=.01,
			random_state=random_state,
			transform_seed=random_state,
			loss_report_frequency=1,
			n_training_epochs=2,
			),
		])
	first_idx = int(len(thdays)*0/10)
	x_train = np.concatenate([days_embeddings[thday] for thday in thdays[max(0, first_idx-1):]], axis=0)
	y_train = np.concatenate([days_y_true[thday] for thday in thdays[max(0, first_idx-1):]], axis=0)
	_, idxs = np.unique(x_train, return_index=True, axis=0) # drop duplicated
	x_train = x_train[idxs]
	y_train = y_train[idxs]
	print('fit')
	dim_reductor.fit(x_train,
		reduction_map_kwargs={'y':y_train} if supervised else None,
		)
	print('transform')
	days_dim_reductions = {thday:dim_reductor.transform(days_embeddings[thday]) for thday in thdays}

	results = {
		'model_name':train_handler.model.get_name(),
		'survey':dataset.survey,
		'band_names':dataset.band_names,
		'class_names':dataset.class_names,
		'lcobj_names':dataset.get_lcobj_names(),

		'thdays':thdays,
		'days_dim_reductions':days_dim_reductions,
		'days_y_true':days_y_true,
	}

	### save file
	save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}~model_enc_key={model_enc_key}.d'
	files.save_pickle(save_filedir, results) # save file
	dataset.reset_max_day() # very important!!
	dataset.calcule_precomputed() # very important!!
	return