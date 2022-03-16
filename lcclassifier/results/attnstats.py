from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as strings
import matplotlib.pyplot as plt
import matplotlib
from . import utils as utils
from fuzzytools.lists import flat_list
import fuzzytools.matplotlib.ax_styles as ax_styles
from fuzzytools.strings import latex_bf_alphabet_count
from fuzzytools.matplotlib.utils import save_fig

RANDOM_STATE = None
LEN_TH = 3
CLASSES_STYLES = _C.CLASSES_STYLES
FIGSIZE = (14, 8)
DPI = 200
COLOR_DICT = _C.COLOR_DICT

###################################################################################################################################################

def plot_slope_distance_attnstats(rootdir, cfilename, kf, lcset_name, model_names,
	train_mode='pre-training',
	figsize=FIGSIZE,
	dpi=DPI,
	attn_th=0.5,
	len_th=LEN_TH,
	n_bins=50,
	bins_xrange=[None, None],
	bins_yrange=[None, None],
	dj=3,
	distance_mode='mean', # local mean median
	linewidth=.75,
	cmap_name='inferno_r', # viridis_r inferno_r coolwarm_r
	guide_color='k', # w k
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/attnstats/{cfilename}'
		if not ftfiles.path_exists(load_roodir):
			continue
		files, files_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			imbalanced_kf_mode='ignore', # error oversampling ignore
			random_state=RANDOM_STATE,
			)
		print(f'{model_name} {files_ids}({len(files_ids)}#)')
		assert len(files)>0

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		#days = files[0]()['days']

		target_class_names = class_names
		x_key = f'peak_distance.j~dj={dj}~mode={distance_mode}'
		y_key = f'local_slope_m.j~dj={dj}'
		label_dict = {
			x_key:f'SN-peak-distance [days]',
			y_key:f'SN-local-slope',
			}

		fig, axs = plt.subplots(2, len(band_names), figsize=figsize, dpi=dpi)
		for kb,b in enumerate(band_names):
			xy_marginal = []
			xy_attn = []
			attn_scores_collections = flat_list([f()['attn_scores_collection'][b] for f in files])
			for attn_scores_collection in attn_scores_collections:
				c = attn_scores_collection['c']
				b_len = attn_scores_collection['b_len']
				lc_features = attn_scores_collection['lc_features']
				for d in lc_features:
					xy_marginal += [[d[x_key], d[y_key]]]
					if d['attn_scores_min_max_k.j']>=attn_th and b_len>=len_th and c in target_class_names:
						xy_attn += [[d[x_key], d[y_key]]]
		
			xy_marginal = np.array(xy_marginal)
			xy_attn = np.array(xy_attn)
			print('xy_marginal', xy_marginal.shape, 'xy_attn', xy_attn.shape)

			xrange0 = xy_attn[:,0].min() if bins_xrange[0] is None else bins_xrange[0]
			xrange1 = xy_attn[:,0].max() if bins_xrange[1] is None else bins_xrange[1]
			yrange0 = xy_attn[:,1].min() if bins_yrange[0] is None else bins_yrange[0]
			yrange1 = xy_attn[:,1].max() if bins_yrange[1] is None else bins_yrange[1]

			d = {
				'xy_marginal':{'xy':xy_marginal, 'title':'Joint distribution'},
				'xy_attn':{'xy':xy_attn, 'title':f'Conditional joint distribution using '+'$\\bar{s}\\geq'+str(attn_th)+'$'},
				}
			for kax,xy_name in enumerate(['xy_marginal', 'xy_attn']):
				ax = axs[kax,kb]
				xy = d[xy_name]['xy']
				H, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=(np.linspace(xrange0, xrange1, n_bins), np.linspace(yrange0, yrange1, n_bins)))
				H = H.T  # Let each row list bins with common y range.
				extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
				H = np.ma.filled(np.ma.masked_where((H <= 0), H), np.nan) # cleans plot
				cmap = getattr(matplotlib.cm, cmap_name)
				cmap.set_bad('w', alpha=1)
				ax.imshow(H, interpolation='nearest', origin='lower', aspect='auto',
					cmap=cmap, 
					extent=extent,
					)
				ax.axvline(0, linewidth=linewidth, color=guide_color)
				ax.axhline(0, linewidth=linewidth, color=guide_color)
				title = ''
				title += f'{latex_bf_alphabet_count(kb, kax)}{d[xy_name]["title"]}; band={b}'+'\n'
				ax.set_title(title[:-1])

				txt_x = 1
				txt_y = yedges[0]*0.99
				ax.text(-txt_x, txt_y, 'pre SN-peak\nregion', fontsize=12, c=guide_color, ha='right', va='bottom')
				ax.text(txt_x, txt_y, 'post SN-peak\nregion', fontsize=12, c=guide_color, ha='left', va='bottom')
				ax_styles.set_color_borders(ax, COLOR_DICT[b])

				xlabel = label_dict[x_key]
				ylabel = label_dict[y_key]
				if kb==0:
					ax.set_ylabel(ylabel)
					ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
					ax.yaxis.major.formatter._useMathText = True
				else:
					ax.set_yticklabels([])
				if kax==0:
					ax.set_xticklabels([])
				else:
					ax.set_xlabel(xlabel)

		suptitle = ''
		# suptitle += f''+'\n'
		suptitle += f'{utils.get_fmodel_name(model_name)}'+'\n'
		# suptitle += f'set={survey} [{lcset_name.replace(".@", "")}]'+'\n'
		fig.suptitle(suptitle[:-1], va='top', y=1.01)
		fig.tight_layout()
		save_fig(fig, f'../temp/exp=attnstats~mdl={model_name}.pdf', closes_fig=False)
		plt.show()