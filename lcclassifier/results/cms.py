from __future__ import print_function
from __future__ import division

import numpy as np
import fuzzytools.files as ftfiles
import matplotlib.pyplot as plt
from fuzzytools.datascience.xerror import XError
from fuzzytools.datascience.cms import ConfusionMatrix
from . import utils as utils
from fuzzytools.matplotlib.cm_plots import plot_custom_confusion_matrix
from fuzzytools.matplotlib.animators import PlotAnimator
from fuzzytools.progress_bars import ProgressBar
from fuzzytools.strings import latex_bf_alphabet_count
from fuzzytools.matplotlib.utils import save_fig
from . import _C


FIGSIZE = (6, 5)
DPI = 200
RANDOM_STATE = None
NEW_ORDER_CLASS_NAMES = ['SNIa', 'SNIbc', 'SNII*', 'SLSN']
DICT_NAME = 'thdays_class_metrics'


def plot_cm(rootdir, cfilename, kf, lcset_name, model_names,
            figsize=FIGSIZE,
            dpi=DPI,
            alphabet_count_offset=1,
            train_mode='fine-tuning',
            export_animation=False,
            new_order_class_names=NEW_ORDER_CLASS_NAMES,
            dict_name=DICT_NAME,
            ):
    for kmn, model_name in enumerate(model_names):
        load_roodir = f'{rootdir}/{model_name}/{train_mode}/performance/{cfilename}'
        files, file_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
                                                             fext='d',
                                                             imbalanced_kf_mode='oversampling',  # error oversampling
                                                             random_state=RANDOM_STATE,
                                                             )
        print(f'ids={file_ids}(n={len(file_ids)}#); model={model_name}')
        if len(files) == 0:
            continue
        survey = files[0]()['survey']
        band_names = files[0]()['band_names']
        class_names = files[0]()['class_names']
        is_parallel = 'Parallel' in model_name
        thdays = files[0]()['thdays']

        save_filename = f'../temp/exp=cm~model={model_name}'
        plot_animator = PlotAnimator(f'{save_filename}.gif',
                                     is_dummy=not export_animation,
                                     save_frames=True,
                                     )
        thdays = thdays if export_animation else [thdays[-1]]
        bar = ProgressBar(len(thdays))
        for kd, thday in enumerate(thdays):
            bar(f'thday={thday:.3f} [days]')
            brecall_xe = XError([f()[f'{dict_name}_df'].loc[f()[f'{dict_name}_df']['_thday'] == thday]['b-recall'].item() for f in files])
            bf1score_xe = XError([f()[f'{dict_name}_df'].loc[f()[f'{dict_name}_df']['_thday'] == thday]['b-f1score'].item() for f in files])
            baucroc_xe = XError([f()[f'{dict_name}_df'].loc[f()[f'{dict_name}_df']['_thday'] == thday]['b-aucroc'].item() for f in files])
            for k in range(0, len(brecall_xe)):
                print(f'file_id={file_ids[k]}; b-recall={brecall_xe.get_item(k)}; b-f1score={bf1score_xe.get_item(k)}; b-aucroc={baucroc_xe.get_item(k)};')
                
            cm = ConfusionMatrix([f()['thdays_cm'][thday] for f in files], class_names)
            cm.reorder_classes(new_order_class_names)
            for c in new_order_class_names:
                if 'TimeModAttn' in model_name:
                    sp_text = 'p' if is_parallel else 's'
                    print(cm.get_diagonal_dict()[c].get_raw_repr(f'{sp_text}_{c}_tp'))
                pass
            alphabet_count = 0 if 'Serial' in model_name else 1
            title = ''
            title += f'{latex_bf_alphabet_count(alphabet_count+alphabet_count_offset)}{utils.get_fmodel_name(model_name)}\n'
            title += f'b-Recall={brecall_xe}; b-$F_1$score={bf1score_xe}\n'
            title += f'th-day={thday:.0f} [days]\n'
            fig, ax = plot_custom_confusion_matrix(cm,
                                                   title=title[:-1],
                                                   figsize=figsize,
                                                   dpi=dpi,
                                                   true_label_d={c: f'({k}#)' for c, k in zip(class_names, np.sum(files[0]()['thdays_cm'][thday], axis=1))},
                                                   lambda_c=lambda x: x.replace('*', ''),
                                                   )
            uses_close_fig = kd < len(thdays) - 1
            plot_animator.append(fig, uses_close_fig)

        bar.done()
        plot_animator.save()
        save_fig(fig, f'{save_filename}.pdf', closes_fig=0)
        plt.show()
