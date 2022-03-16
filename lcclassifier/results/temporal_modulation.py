from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
from matplotlib import cm
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import fuzzytools.matplotlib.ax_styles as ax_styles
from . import utils as utils
from fuzzytools.strings import latex_bf_alphabet_count
from fuzzytools.matplotlib.utils import save_fig

RANDOM_STATE = _C.RANDOM_STATE
PERCENTILE_PLOT = 95
SHADOW_ALPHA = .25
FIGSIZE = (14, 8)
DPI = 200
COLOR_DICT = _C.COLOR_DICT

### hardcoded
EMPIRICAL_PEAKDAY = {
	'r':13.04,
	'g':11.00,
	}
EMPIRICAL_LASTDAY = {
	'r':52.84,
	'g':39.98,
	}

###################################################################################################################################################

def _get_diff(time, x):
	assert len(time.shape)==1
	assert len(x.shape)==1
	new_x = np.diff(x)/np.diff(time)
	new_x = np.concatenate([[np.nan], new_x], axis=0)
	return new_x

def get_diff(time, x,
	degree=1,
	):
	new_x = copy(x)
	for _ in range(0, degree):
		new_x = _get_diff(time, new_x)
	return new_x

###################################################################################################################################################

def plot_temporal_modulation(rootdir, cfilename, kf, lcset_name, model_names,
	train_mode='pre-training',
	layers=1,
	figsize=FIGSIZE,
	dpi=DPI,
	n=1e4,
	percentile=PERCENTILE_PLOT,
	shadow_alpha=SHADOW_ALPHA,
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/temporal_encoding/{cfilename}'
		if not ftfiles.path_exists(load_roodir):
			continue
		files, files_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			imbalanced_kf_mode='ignore', # error oversampling ignore
			random_state=RANDOM_STATE,
			)
		print(f'{model_name} {files_ids}({len(files_ids)}#)')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		mn_dict = strings.get_dict_from_string(model_name)
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl
		if not is_parallel:
			continue

		first_day = files[0]()['days'][0]
		# last_day = files[0]()['days'][-1]
		last_day = 100

		global_median_curves_d = {}
		fig, axs = plt.subplots(2, len(band_names), figsize=figsize, dpi=dpi)
		for kfile,file in enumerate(files):
			for kb,b in enumerate(band_names):
				time = np.linspace(first_day, last_day, int(n)) # (t)
				d = file()['temporal_encoding_info']['encoder'][f'ml_attn.{b}']['te_film']
				weight = d['weight'].T # (2K,2M)>(2M,2K)
				te_ws = d['te_ws'] # (2M)
				te_phases = d['te_phases'] # (2M)
				te_scales = d['te_scales'] # (2M)
				sin_arg = te_ws[None,None,:]*time[None,:,None]+te_phases[None,None,:] # (1,t,2M)
				encoding = te_scales[None,None,:]*np.sin(sin_arg) # (1,t,2M)
				alphas, betas = np.split(encoding@weight, 2, axis=-1) # (1,t,2M)@(2M,2K)>(1,t,2K)>(1,t,K),(1,t,K)
				scales = []
				biases = []
				for kfu in range(0, alphas.shape[-1]):
					dalpha = get_diff(time, alphas[0,:,kfu])**2 # (t)
					dbeta = get_diff(time, betas[0,:,kfu])**2 # (t)
					scales += [dalpha]
					biases += [dbeta]

				d = {
					'scale':{'curve':scales, 'c':'r'},
					'bias':{'curve':biases, 'c':'g'},
					}

				for kax,curve_name in enumerate(['scale', 'bias']):
					ax = axs[kax,kb]
					curves = d[curve_name]['curve']
					c = 'k'
					median_curve = np.mean(np.concatenate([curve[None]for curve in curves], axis=0), axis=0) # median mean
					if not f'{kax}/{kb}/{b}' in global_median_curves_d.keys():
						global_median_curves_d[f'{kax}/{kb}/{b}'] = []
					global_median_curves_d[f'{kax}/{kb}/{b}'] += [median_curve]
					ax.plot(time, median_curve,
						c=c,
						alpha=1,
						lw=.5,
						)
					ax.plot([None], [None], c=c, label=f'variability time-function (one model run)' if kfile==0 else None)
					ax.legend(loc='upper right')
					ax.grid(alpha=.0)
					title = f'{latex_bf_alphabet_count(kb, kax)} {curve_name.capitalize()} variability time-function; band={b}'+'\n'
					ax.set_title(title[:-1])
					ax_styles.set_color_borders(ax, COLOR_DICT[b])
					if kb==0:
						ax.set_ylabel(f'variability')
					else:
						pass
					if kax==0:
						ax.set_xticklabels([])
					else:
						ax.set_xlabel(f'time [days]')
					# break

			model_label = utils.get_fmodel_name(model_name)
			suptitle = ''
			suptitle += f'{model_label}'+'\n'
			fig.suptitle(suptitle[:-1], va='top', y=1.01)

		for k in global_median_curves_d.keys():
			kax,kb,b = k.split('/')
			median_curves = global_median_curves_d[k]
			ax = axs[int(kax), int(kb)]
			ax.plot(time, np.median(np.concatenate([median_curve[None] for median_curve in median_curves], axis=0), axis=0), '-', c='w', lw=3.5)
			ax.plot(time, np.median(np.concatenate([median_curve[None] for median_curve in median_curves], axis=0), axis=0), '-', c='r',
				label=f'median variability time-function (median of all model runs)',
				)
			ax.axvline(EMPIRICAL_PEAKDAY[b], linestyle='--', c='k', label='SN-peak time (empirical dataset median)')
			ax.axvspan(EMPIRICAL_LASTDAY[b], last_day, alpha=0.1, color='k', lw=0, label=f'post SN last observation-time region (empirical dataset median)')
			ax.legend(loc='upper right')
			ax.set_ylim((0, None))
			ax.set_xlim((first_day, last_day))
			ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
			ax.yaxis.major.formatter._useMathText = True

		fig.tight_layout()
		save_fig(fig, f'../temp/exp=tmod~mdl={model_name}.pdf', closes_fig=False)
		plt.show()