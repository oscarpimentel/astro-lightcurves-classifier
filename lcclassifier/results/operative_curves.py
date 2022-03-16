from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
import fuzzytools.matplotlib.fills as fills
import matplotlib.pyplot as plt
from fuzzytools.matplotlib.utils import save_fig
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from matplotlib import cm
from lchandler._C import CLASSES_STYLES
from fuzzytools.strings import latex_bf_alphabet_count

RANDOM_STATE = 0
PERCENTILE = 75
SHADOW_ALPHA = 0.1
FIGSIZE = (7, 6)
FIGSIZE_2X1 = (14, 8)
DPI = 200
BASELINE_ROOTDIR = _C.BASELINE_ROOTDIR
BASELINE_MODEL_NAME = _C.BASELINE_MODEL_NAME
METRICS_D = _C.METRICS_D
DICT_NAME = 'thdays_class_metrics'
TRAIN_MODE = 'fine-tuning'

XLABEL_DICT = {
	'rocc':'fpr',
	'prc':'recall',
	}

YLABEL_DICT = {
	'rocc':'tpr',
	'prc':'precision',
	}

AXIS_XLABEL_DICT = {
	'rocc':'FPR',
	'prc':'Recall',
	}

AXIS_YLABEL_DICT = {
	'rocc':'TPR',
	'prc':'Precision',
	}

GUIDE_CURVE_DICT = {
	'rocc':[0,1],
	'prc':[1,0],
	}

################################################################################################################

def plot_ocurve_classes(rootdir, cfilename, kf, lcset_name, model_names, target_classes, thday,
	baselines_dict={},
	figsize=FIGSIZE,
	dpi=DPI,
	train_mode=TRAIN_MODE,
	percentile=PERCENTILE,
	shadow_alpha=SHADOW_ALPHA,
	ocurve_name='rocc',
	baseline_rootdir=BASELINE_ROOTDIR,
	dict_name=DICT_NAME,
	):
	for kmn,model_name in enumerate(model_names):
		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
		files, file_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			imbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		print(f'{file_ids}({len(file_ids)}#); model={model_name}')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		is_parallel = 'Parallel' in model_name
		thdays = files[0]()['thdays']

		for target_class in target_classes:
			xe_aucroc = XError([f()[f'{dict_name}_cdf'][target_class].loc[f()[f'{dict_name}_cdf'][target_class]['_thday']==thday]['aucroc'].item() for f in files])
			label = f'{target_class.replace("*", "")}; AUC={xe_aucroc}'
			color = CLASSES_STYLES[target_class]['c']
			sp_text = 'p' if is_parallel else 's'
			print(xe_aucroc.get_raw_repr(f'{sp_text}_{target_class}_auc'))

			ocurves = [f()[f'{dict_name}_cdf'][target_class].loc[f()[f'{dict_name}_cdf'][target_class]['_thday']==thday][ocurve_name].item() for f in files]
			fills.fill_beetween_percentile(ax, [ocurve[XLABEL_DICT[ocurve_name]] for ocurve in ocurves], [ocurve[YLABEL_DICT[ocurve_name]] for ocurve in ocurves],
				median_kwargs={'color':color, 'alpha':1, 'label':label,},
				fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0,},
				percentile=percentile,
				)

		### baseline
		if not baseline_rootdir is None:
			files, file_ids, kfs = ftfiles.gather_files_by_kfold(baseline_rootdir, kf, lcset_name,
				fext='d',
				imbalanced_kf_mode='oversampling', # error oversampling
				random_state=RANDOM_STATE,
				)
			print(f'{file_ids}({len(file_ids)}#); model={BASELINE_MODEL_NAME}')
			
			for target_class in target_classes:
				xe_aucroc = XError([f()[f'{dict_name}_cdf'][target_class].loc[f()[f'{dict_name}_cdf'][target_class]['_thday']==thday]['auc'+ocurve_name[:-1]].item() for f in files])
				label = f'{target_class.replace("*", "")}; AUC={xe_aucroc} (BRF)'
				color = CLASSES_STYLES[target_class]['c']
				print(xe_aucroc.get_raw_repr(f'brf_{target_class}_auc'))

				ocurves = [f()[f'{dict_name}_cdf'][target_class].loc[f()[f'{dict_name}_cdf'][target_class]['_thday']==thday][ocurve_name].item() for f in files]
				fills.fill_beetween_percentile(ax, [ocurve[XLABEL_DICT[ocurve_name]] for ocurve in ocurves], [ocurve[YLABEL_DICT[ocurve_name]] for ocurve in ocurves],
					median_kwargs={'color':color, 'alpha':1, 'linestyle':'--', 'label':label,},
					fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0,},
					percentile=percentile,
					)

		ax.plot([0, 1], GUIDE_CURVE_DICT[ocurve_name], '--', color='k', alpha=1, lw=1)
		ax.set_xlabel(AXIS_XLABEL_DICT[ocurve_name])
		ax.set_ylabel(AXIS_YLABEL_DICT[ocurve_name])
		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.0)
		ax.grid(alpha=.0, zorder=-1.0)
		ax.legend(loc='lower right')

		title = ''
		# title += f'{ocurve_name.upper()[:-1]} operative curves for SNe classes'+'\n'
		title += f'{latex_bf_alphabet_count(0 if "Serial" in model_name else 1)}{utils.get_fmodel_name(model_name)}'+'\n'
		# title += f'set={survey} [{lcset_name.replace(".@", "")}]'+'\n'
		title += f'th-day={thday:.0f} [days]'+'\n'
		ax.set_title(title[:-1])

	fig.tight_layout()
	save_fig(fig, f'../temp/exp=roccs~model={model_name}.pdf', closes_fig=False)
	plt.show()