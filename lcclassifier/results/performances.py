from __future__ import print_function
from __future__ import division

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.matplotlib.fills as fills
import matplotlib.pyplot as plt
from fuzzytools.matplotlib.lims import AxisLims
from fuzzytools.datascience.xerror import XError
from . import utils as utils
from fuzzytools.strings import latex_bf_alphabet_count
from fuzzytools.matplotlib.utils import save_fig
from . import _C


RANDOM_STATE = None
STD_PROP = 1
SHADOW_ALPHA = .1
FIGSIZE = (14, 4.5)
DPI = 200
BASELINE_D = _C.BASELINE_D
METRICS_D = _C.METRICS_D
COLOR_DICT = _C.COLOR_DICT
FILL_USES_MEAN = True
DICT_NAME = 'thdays_class_metrics'
N_XTICKS = 4


def plot_metric(rootdir, cfilename, kf, lcset_name, model_names, metric_name,
                target_class=None,
                figsize=FIGSIZE,
                dpi=DPI,
                train_mode='fine-tuning',
                std_prop=STD_PROP,
                shadow_alpha=SHADOW_ALPHA,
                baseline_d=BASELINE_D,
                dict_name=DICT_NAME,
                n_xticks=N_XTICKS,
                ):
    new_metric_name = 'b-' + METRICS_D[metric_name]['mn'] if target_class is None else METRICS_D[metric_name]['mn'] + '$_\\mathregular{(' + target_class + ')}$'
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axis_lims = AxisLims({'x': (None, None), 'y': (0, 1)}, {'x': .0, 'y': .05})
    sp_model_names = utils.get_sorted_model_names(model_names, merged=False)
    # plot deep learning models
    for kax, ax in enumerate(axs):
        if len(sp_model_names[kax]) == 0:
            continue
        for kmn, model_name in enumerate(sp_model_names[kax]):
            load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
            files, file_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
                                                                 fext='d',
                                                                 imbalanced_kf_mode='oversampling',  # error oversampling
                                                                 random_state=RANDOM_STATE,
                                                                 )
            # print(f'{file_ids}({len(file_ids)}#); model={model_name}')
            if len(files) == 0:
                continue

            survey = files[0]()['survey']
            band_names = files[0]()['band_names']
            class_names = files[0]()['class_names']
            thdays = files[0]()['thdays']

            if target_class is None:
                metric_curves = [f()[f'{dict_name}_df'][f'b-{metric_name}'].values for f in files]
            else:
                metric_curves = [f()[f'{dict_name}_cdf'][target_class][f'{metric_name}'].values for f in files] 

            fmodel_name = utils.get_fmodel_name(model_name)
            color = utils.get_model_color(model_name, model_names)
            linestyle = '-'
            ax, new_x, median_y, yrange = fills.fill_beetween_mean_std(ax, [thdays for _ in metric_curves], metric_curves,
                                                                       mean_kwargs={'color': color, 'alpha': 1, 'linestyle': linestyle},
                                                                       fill_kwargs={'color': color, 'alpha': shadow_alpha, 'lw': 0},
                                                                       returns_extras=True,
                                                                       std_prop=std_prop,
                                                                       )

            xe_metric_curve_auc = XError(np.mean(np.concatenate([metric_curve[None] for metric_curve in metric_curves], axis=0), axis=-1))  # (b,t)
            label = f'{fmodel_name}; mtdCA={xe_metric_curve_auc}'
            ax.plot([None], [None], color=color, linestyle=linestyle, label=label)

            axis_lims.append('x', thdays)
            axis_lims.append('y', np.array(yrange))

        suptitle = ''
        suptitle += f'{new_metric_name} curve using the moving threshold-day\n'
        fig.suptitle(suptitle[:-1], va='top', y=1.01)

    # plot baselines
    if baseline_d is not None:
        for kax, ax in enumerate(axs):
            for model_name in baseline_d.keys():
                baseline_path = baseline_d[model_name]['path']
                files, file_ids, kfs = ftfiles.gather_files_by_kfold(baseline_path, kf, lcset_name,
                                                                     fext='d',
                                                                     imbalanced_kf_mode='oversampling',  # error oversampling
                                                                     random_state=RANDOM_STATE,
                                                                     )
                # print(f'{file_ids}({len(file_ids)}#)')
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
                if '[r]' in model_name:
                    color = (.4,) * 3
                    marker = 'D'
                elif '[s]' in model_name:
                    color = (.0,) * 3
                    marker = 'o'
                elif '[r+r]' in model_name:
                    color = (.2,) * 3
                    marker = 's'
                else:
                    raise NotImplementedError(f'model_name={model_name}')
                zorder = 2
                label = fmodel_name = utils.get_fmodel_name(model_name)
                mean_kwargs = {'color': color, 'alpha': 1, 'marker': marker, 'markersize': 5, 'markeredgecolor': color, 'markerfacecolor': 'None', 'markevery': 0.05, 'zorder': zorder, 'label': label}
                fill_kwargs = {'color': color, 'alpha': shadow_alpha, 'lw': 0, 'zorder': zorder}
                ax, new_x, median_y, yrange = fills.fill_beetween_mean_std(ax, thdays_computed_curves, metric_curves,
                                                                           mean_kwargs=mean_kwargs,
                                                                           fill_kwargs=fill_kwargs,
                                                                           returns_extras=True,
                                                                           std_prop=std_prop,
                                                                           )
                argmax_median_y = np.argmax(median_y)
                max_median_y = median_y[argmax_median_y]
                ax.axhline(max_median_y,
                           linestyle=':',
                           c=color,
                           label=f'{new_metric_name}={max_median_y:.3f}',
                           zorder=zorder,
                           lw=1,
                           )
                ax.axvline(new_x[0],
                           linestyle='--',
                           c=color,
                           label=f'threshold-day={new_x[0]:.0f} [days]',
                           zorder=zorder,
                           lw=1,
                           )

            ax.set_xticks(thdays[::-1][::n_xticks][::-1])
            ax.set_xticklabels([f'{xt:.0f}' for xt in ax.get_xticks()], rotation=90)
            ax.set_xlabel('threshold-day [days]')
            if kax == 0:
                ax.set_ylabel(new_metric_name)
                ax.set_title(f'{latex_bf_alphabet_count(0)}Models with serial encoder')
            else:
                ax.set_yticklabels([])
                ax.set_title(f'{latex_bf_alphabet_count(1)}Models with parallel encoder')

            axis_lims.set_ax_axis_lims(ax)
            ax.legend(loc='lower right')

    fig.tight_layout()
    save_fig(fig, f'temp/exp=performance_curve~metric_name={metric_name}.pdf', closes_fig=0)
    plt.show()
