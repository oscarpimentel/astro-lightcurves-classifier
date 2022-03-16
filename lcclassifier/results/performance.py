from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.matplotlib.fills as fills
from fuzzytools.matplotlib.lims import AxisLims
from matplotlib import cm
import matplotlib.pyplot as plt
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from fuzzytools.strings import latex_bf_alphabet_count
from fuzzytools.matplotlib.utils import save_fig

RANDOM_STATE = None
STD_PROP = 1
SHADOW_ALPHA = .1
FIGSIZE = (14, 4.5)
DPI = 200
BASELINE_ROOTDIR = _C.BASELINE_ROOTDIR
BASELINE_MODEL_NAME = _C.BASELINE_MODEL_NAME
METRICS_D = _C.METRICS_D
COLOR_DICT = _C.COLOR_DICT
FILL_USES_MEAN = True
DICT_NAME = 'thdays_class_metrics'
N_XTICKS = 4

###################################################################################################################################################

def plot_metric(rootdir, cfilename, kf, lcset_name, model_names, metric_names,
	target_class=None,
	figsize=FIGSIZE,
	dpi=DPI,
	train_mode='fine-tuning',
	std_prop=STD_PROP,
	shadow_alpha=SHADOW_ALPHA,
	baseline_rootdir=BASELINE_ROOTDIR,
	dict_name=DICT_NAME,
	n_xticks=N_XTICKS,
	):
	for metric_name in metric_names:
		new_metric_name = 'b-'+METRICS_D[metric_name]['mn'] if target_class is None else METRICS_D[metric_name]['mn']+'$_\\mathregular{('+target_class+')}$'
		fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
		axis_lims = AxisLims({'x':(None, None), 'y':(0, 1)}, {'x':.0, 'y':.05})
		sp_model_names = utils.get_sorted_model_names(model_names, merged=False)
		for kax,ax in enumerate(axs):
			if len(sp_model_names[kax])==0:
				continue
			for kmn,model_name in enumerate(sp_model_names[kax]):
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
				thdays = files[0]()['thdays']

				if target_class is None:
					metric_curves = [f()[f'{dict_name}_df'][f'b-{metric_name}'].values for f in files]
				else:
					metric_curves = [f()[f'{dict_name}_cdf'][target_class][f'{metric_name}'].values for f in files] 

				fmodel_name, mn_dict = utils.get_fmodel_name(model_name, returns_mn_dict=True)
				color = utils.get_model_color(model_name, model_names)
				linestyle = '-'
				ax, new_x, median_y, yrange = fills.fill_beetween_mean_std(ax, [thdays for _ in metric_curves], metric_curves,
					mean_kwargs={'color':color, 'alpha':1, 'linestyle':linestyle},
					fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0,},
					returns_extras=True,
					std_prop=std_prop,
					)

				xe_metric_curve_auc = XError(np.mean(np.concatenate([metric_curve[None] for metric_curve in metric_curves], axis=0), axis=-1)) # (b,t)
				label = f'{fmodel_name}; mtdCA={xe_metric_curve_auc}'
				ax.plot([None], [None], color=color, linestyle=linestyle, label=label)

				axis_lims.append('x', thdays)
				axis_lims.append('y', np.array(yrange))

			suptitle = ''
			suptitle += f'{new_metric_name} curve using the moving threshold-day'+'\n'
			fig.suptitle(suptitle[:-1], va='top', y=1.01)

		for kax,ax in enumerate(axs):
			if not baseline_rootdir is None:
				files, file_ids, kfs = ftfiles.gather_files_by_kfold(baseline_rootdir, kf, lcset_name,
					fext='d',
					imbalanced_kf_mode='oversampling', # error oversampling
					random_state=RANDOM_STATE,
					)
				print(f'{file_ids}({len(file_ids)}#); model={BASELINE_MODEL_NAME}')
				thdays = files[0]()['thdays']
				thdays_computed_curves = []
				metric_curves = []
				for f in files:
					thdays_computed_curves += [np.array(f()['thdays_computed'])]
					if target_class is None:
						metric_curve = f()[f'{dict_name}_df'][f'b-{metric_name}'].values
						metric_curves += [metric_curve]
					else:
						metric_curve = f()[f'{dict_name}_cdf'][target_class][f'{metric_name}'].values
						metric_curves += [metric_curve]
				
				color = 'k'
				label = f'{utils.get_fmodel_name(BASELINE_MODEL_NAME)}'
				ax, new_x, median_y, yrange = fills.fill_beetween_mean_std(ax, thdays_computed_curves, metric_curves,
					mean_kwargs={'color':color, 'alpha':1, 'marker':'D', 'markersize':0, 'markerfacecolor':'None', 'markevery':[0], 'zorder':-1, 'label':label},
					fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0, 'zorder':-1},
					returns_extras=True,
					std_prop=std_prop,
					)
				argmax_median_y = np.argmax(median_y)
				max_median_x = new_x[argmax_median_y]
				max_median_y = median_y[argmax_median_y]
				ax.axhline(max_median_y,
					linestyle=':',
					c='k',
					label=f'{new_metric_name}={max_median_y:.3f}',
					zorder=-1,
					lw=1,
					)
				ax.axvline(new_x[0],
					linestyle='--',
					c='k',
					label=f'threshold-day={new_x[0]:.0f} [days]',
					zorder=-1,
					lw=1,
					)

			ax.set_xticks(thdays[::-1][::n_xticks][::-1])
			ax.set_xticklabels([f'{xt:.0f}' for xt in ax.get_xticks()], rotation=90)
			ax.set_xlabel('threshold-day [days]')
			if kax==0:
				ax.set_ylabel(new_metric_name)
				ax.set_title(f'{latex_bf_alphabet_count(0)}Models with serial encoder')
			else:
				ax.set_yticklabels([])
				ax.set_title(f'{latex_bf_alphabet_count(1)}Models with parallel encoder')

			axis_lims.set_ax_axis_lims(ax)
			ax.legend(loc='lower right')

		fig.tight_layout()
		save_fig(fig, f'../temp/exp=performance_curve~metric_name={metric_name}.pdf', closes_fig=0)
		plt.show()