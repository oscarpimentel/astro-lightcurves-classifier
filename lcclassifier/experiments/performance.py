from __future__ import print_function
from __future__ import division
from . import _C

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy, minibatch_dict_collate
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
import fuzzytools.datascience.metrics as fcm
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts
import fuzzytorch.models.seq_utils as seq_utils
import pandas as pd

DEFAULT_MIN_DAY = _C.DEFAULT_MIN_DAY
DEFAULT_DAYS_N = _C.DEFAULT_DAYS_N
REC_LOSS_K = _C.REC_LOSS_K
REC_LOSS_EPS = _C.REC_LOSS_EPS

###################################################################################################################################################

def save_performance(train_handler, data_loader, save_rootdir,
	target_is_onehot:bool=False,
	true_y_key='target/y',
	pred_y_key='model/y',
	days_n:int=DEFAULT_DAYS_N,
	**kwargs):
	train_handler.load_model(keys_to_change_d=_C.KEYS_TO_CHANGE_D) # important, refresh to best model
	train_handler.model.eval() # important, model eval mode
	dataset = data_loader.dataset # get dataset
	dataset.reset_max_day() # always reset max day

	thdays_rec_metrics_df = DFBuilder()
	thdays_predictions = {}
	thdays_class_metrics_df = DFBuilder()
	thdays_class_metrics_cdf = {c:DFBuilder() for c in dataset.class_names}
	thdays_cm = {}
	thdays_class_metrics_all_bands_df = DFBuilder()
	thdays_class_metrics_all_bands_cdf = {c:DFBuilder() for c in dataset.class_names}

	thdays = np.linspace(DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
	bar = ProgressBarMulti(len(thdays), 4)
	with torch.no_grad():
		can_be_in_loop = True
		for thday in thdays:
			dataset.set_max_day(thday) # very important!!
			dataset.calcule_precomputed() # very important!!
			try:
				if can_be_in_loop:
					tdicts = []
					for ki,in_tdict in enumerate(data_loader):
						_tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device))
						tdicts += [_tdict]
					tdict = minibatch_dict_collate(tdicts)

					### wmse
					wmse_loss_bdict = {}
					for kb,b in enumerate(dataset.band_names):
						p_onehot = tdict[f'input/onehot.{b}'][...,0] # (n,t)
						#p_rtime = tdict[f'input/rtime.{b}'][...,0] # (n,t)
						#p_dtime = tdict[f'input/dtime.{b}'][...,0] # (n,t)
						#p_x = tdict[f'input/x.{b}'] # (n,t,i)
						p_rerror = tdict[f'target/rerror.{b}'] # (n,t,1)
						p_recx = tdict[f'target/recx.{b}'] # (n,t,1)
						p_decx = tdict[f'model/decx.{b}'] # (n,t,1)

						p_wmse_loss = (p_recx-p_decx)**2/(REC_LOSS_K*(p_rerror**2)+REC_LOSS_EPS) # (n,t,1)
						p_wmse_loss = seq_utils.seq_avg_pooling(p_wmse_loss, p_onehot)[...,0] # (n,t,1)>(n,t)>(n)
						wmse_loss_bdict[b] = p_wmse_loss # (n)

					wmse_loss = torch.cat([wmse_loss_bdict[b][...,None] for b in dataset.band_names], axis=-1).mean(dim=-1) # (n,b)>(n)
					wmse_loss = wmse_loss.mean() # (n)>()

					thdays_rec_metrics_df.append(thday, {
						'_thday':thday,
						'wmse':tensor_to_numpy(wmse_loss),
						})

					### class prediction
					y_true = tdict[true_y_key] # (n)
					y_pred_p = torch.nn.functional.softmax(tdict[pred_y_key], dim=-1) # (n,c)
					if target_is_onehot:
						assert y_pred_.shape==y_true.shape
						y_true = torch.argmax(y_true, dim=-1)

					# metrics
					y_true = tensor_to_numpy(y_true)
					y_pred_p = tensor_to_numpy(y_pred_p)
					thdays_predictions[thday] = {'y_true':y_true, 'y_pred_p':y_pred_p}
					metrics_cdict, metrics_dict, cm = fcm.get_multiclass_metrics(y_pred_p, y_true, dataset.class_names)
					thdays_class_metrics_df.append(thday, update_dicts([{'_thday':thday}, metrics_dict]))
					for c in dataset.class_names:
						thdays_class_metrics_cdf[c].append(thday, update_dicts([{'_thday':thday}, metrics_cdict[c]]))
					
					### confusion matrix
					thdays_cm[thday] = cm

					### progress bar
					recall = {c:metrics_cdict[c]['recall'] for c in dataset.class_names}
					bmetrics_dict = {k:metrics_dict[k] for k in metrics_dict.keys() if 'b-' in k}
					bar([f'lcset_name={dataset.lcset_name}; _thday={thday:.3f}', f'wmse_loss={wmse_loss}', f'bmetrics_dict={bmetrics_dict}', f'recall={recall}'])

					### all bands observed
					s_onehot = tdict[f'input/s_onehot'] # (n,t,b)
					all_bands = torch.all(torch.any(s_onehot, dim=1), dim=-1) # (n,t,b)>(n,t)>(n)
					all_bands = tensor_to_numpy(all_bands)
					all_bands_y_pred_p = y_pred_p[all_bands]
					all_bands_y_true = y_true[all_bands]
					unique_classes = np.unique(all_bands_y_true)
					if len(unique_classes)==len(dataset.class_names):
						metrics_cdict, metrics_dict, cm = fcm.get_multiclass_metrics(all_bands_y_pred_p, all_bands_y_true, dataset.class_names)
						thdays_class_metrics_all_bands_df.append(thday, update_dicts([{'_thday':thday}, metrics_dict]))
						for c in dataset.class_names:
							thdays_class_metrics_all_bands_cdf[c].append(thday, update_dicts([{'_thday':thday}, metrics_cdict[c]]))

			except KeyboardInterrupt:
				can_be_in_loop = False

	bar.done()
	d = {
		'model_name':train_handler.model.get_name(),
		'survey':dataset.survey,
		'band_names':dataset.band_names,
		'class_names':dataset.class_names,
		'lcobj_names':dataset.get_lcobj_names(),

		'thdays':thdays,
		'thdays_rec_metrics_df':thdays_rec_metrics_df.get_df(),
		'thdays_predictions':thdays_predictions,
		'thdays_class_metrics_df':thdays_class_metrics_df.get_df(),
		'thdays_class_metrics_cdf':{c:thdays_class_metrics_cdf[c].get_df() for c in dataset.class_names},
		'thdays_cm':thdays_cm,
		'thdays_class_metrics_all_bands_df':thdays_class_metrics_all_bands_df.get_df(),
		'thdays_class_metrics_all_bands_cdf':{c:thdays_class_metrics_all_bands_cdf[c].get_df() for c in dataset.class_names},
		}

	### save file
	save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}.d'
	files.save_pickle(save_filedir, d) # save file
	dataset.reset_max_day() # very important!!
	dataset.calcule_precomputed() # very important!!
	return