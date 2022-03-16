from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
from fuzzytools.dataframes import DFBuilder
from fuzzytools.datascience.xerror import XError
from . import utils as utils
import pandas as pd
from scipy.interpolate import interp1d

RANDOM_STATE = None
BASELINE_ROOTDIR = _C.BASELINE_ROOTDIR
BASELINE_MODEL_NAME = _C.BASELINE_MODEL_NAME
METRICS_D = _C.METRICS_D
DICT_NAME = 'thdays_class_metrics'

###################################################################################################################################################

def get_ps_performance_df(rootdir, cfilename, kf, set_name, model_names, metric_names,
	target_class=None,
	thday=None,
	train_mode='fine-tuning',
	uses_avg=False,
	baseline_rootdir=BASELINE_ROOTDIR,
	dict_name=DICT_NAME,
	):
	info_df = DFBuilder()
	new_model_names = utils.get_sorted_model_names(model_names)
	new_model_names = [BASELINE_MODEL_NAME]+new_model_names if not baseline_rootdir is None else new_model_names
	for kmn,model_name in enumerate(new_model_names):
		is_baseline = 'BRF' in model_name
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}' if not is_baseline else baseline_rootdir
		# print(load_roodir)
		files, file_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, set_name,
			fext='d',
			imbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		print(len(file_ids))
		# print(f'{file_ids}({len(file_ids)}#); model={model_name}')
		if len(files)==0:
			continue

		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		thdays = files[0]()['thdays']
		thday = thdays[-1] if thday is None else thday

		d = {}
		for metric_name in metric_names:
			new_metric_name = strings.latex_sub_superscript(METRICS_D[metric_name]['mn'],
				subscript=('\\text{'+target_class.replace('SN', '')+'}') if not target_class is None else ' ',
				superscript='\\ddag' if uses_avg else ' ',
				)
			new_metric_name = 'b-'+new_metric_name if target_class is None else new_metric_name

			d[new_metric_name] = []
			if uses_avg:
				if is_baseline:
					d[new_metric_name] = XError([-np.inf])
				else:
					if target_class is None:
						metric_curves = [f()[f'{dict_name}_df'][f'b-{metric_name}'].values for f in files]
					else:
						metric_curves = [f()[f'{dict_name}_cdf'][target_class][f'{metric_name}'].values  for f in files]
					xe_metric_curve_auc = XError(np.mean(np.concatenate([metric_curve[None] for metric_curve in metric_curves], axis=0), axis=-1)) # (b,t)>(b)
					d[new_metric_name] = xe_metric_curve_auc
			else:
				if target_class is None:
					xe_metric = XError([f()[f'{dict_name}_df'].loc[f()[f'{dict_name}_df']['_thday']==thday][f'b-{metric_name}'].item() for f in files])
				else:
					xe_metric = XError([f()[f'{dict_name}_cdf'][target_class].loc[f()[f'{dict_name}_df']['_thday']==thday][f'{metric_name}'].item() for f in files])
				d[new_metric_name] = xe_metric

		index = f'Model={utils.get_fmodel_name(model_name)}'
		info_df.append(index, d)

	return info_df

###################################################################################################################################################

def get_ps_times_df(rootdir, cfilename, method, model_names,
	train_mode='pre-training',
	kf='0',
	set_name='test',
	):
	info_df = DFBuilder()
	new_model_names = utils.get_sorted_model_names(model_names)
	for kmn,model_name in enumerate(new_model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/model_info/{cfilename}'
		print(load_roodir)
		files, file_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, set_name,
			fext='d',
			imbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		print(f'{model_name} {file_ids}({len(file_ids)}#); model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']

		d = {}
		loss_name = 'wmse+binxentropy'
		parameters = [f()['parameters'] for f in files][0]
		d['#p'] = parameters
		d['mbIT [secs]'] = sum([f()['monitors'][loss_name]['time_per_iteration'] for f in files])
		d['mbIT/#p [$\\mu$secs]'] = sum([f()['monitors'][loss_name]['time_per_iteration']/parameters*1e6 for f in files])

		index = f'Model={utils.get_fmodel_name(model_name)}'
		info_df.append(index, d)

	return info_df