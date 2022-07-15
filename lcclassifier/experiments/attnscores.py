from __future__ import print_function
from __future__ import division
from . import _C

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy, minibatch_dict_collate
import numpy as np
import fuzzytools.files as files
from fuzzytools.matplotlib.utils import save_fig
from fuzzytools.matplotlib.animators import PlotAnimator
import matplotlib.pyplot as plt
import fuzzytorch.models.seq_utils as seq_utils
from lchandler.plots.lc import plot_lightcurve
import random
from fuzzytools.strings import latex_bf_alphabet_count
from .utils import check_attn_scores
from matplotlib.ticker import ScalarFormatter


SCORES_LAYER = -1
DEFAULT_MIN_DAY = _C.DEFAULT_MIN_DAY
DEFAULT_DAYS_N_AN = _C.DEFAULT_DAYS_N_AN
COLOR_DICT = _C.COLOR_DICT
MIN_MARKERSIZE = 12
MAX_MARKERSIZE = 30
FIGSIZE = (8, 10)
DPI = 200


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"


def save_attnscores_animation(train_handler, data_loader, save_rootdir,
    m:int=2,
    figsize:tuple=FIGSIZE,
    dpi=DPI,
    nc:int=1,
    **kwargs):
    results = []
    for experiment_id in range(0, m):
        random.seed(experiment_id)
        np.random.seed(experiment_id)
        r = _save_attnscores_animation(train_handler, data_loader, save_rootdir, str(experiment_id),
            figsize,
            dpi,
            nc,
            **kwargs)
        results.append(r)
    return results

def _save_attnscores_animation(train_handler, data_loader, save_rootdir, experiment_id,
    figsize:tuple=FIGSIZE,
    dpi=DPI,
    nc:int=1,
    alpha=0.333,
    days_n:int=DEFAULT_DAYS_N_AN,
    animation_duration=10,
    **kwargs):
    train_handler.load_model(keys_to_change_d=_C.KEYS_TO_CHANGE_D) # important, refresh to best model
    train_handler.model.eval() # important, model eval mode
    dataset = data_loader.dataset # get dataset
    
    is_parallel = 'Parallel' in train_handler.model.get_name()
    if not is_parallel:
        return

    plot_animator = PlotAnimator(f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}~exp_id={experiment_id}.gif',
        animation_duration=animation_duration,
        save_frames=True,
        )
    days = np.linspace(DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
    with torch.no_grad():
        lcobj_names = dataset.get_random_stratified_lcobj_names(nc)
        xlims = {lcobj_name: None for lcobj_name in lcobj_names}
        ylims = {lcobj_name: None for lcobj_name in lcobj_names}
        for kday,day in enumerate(days[::-1]): # along days
            dataset.set_max_day(day)
            dataset.calcule_precomputed()

            fig, axs = plt.subplots(len(lcobj_names), 1, figsize=figsize, dpi=dpi)
            for k,lcobj_name in enumerate(lcobj_names):
                ax = axs[k]
                in_tdict, lcobj = dataset.get_item(lcobj_name)
                in_tdict = dataset.fix_tdict(in_tdict)
                train_handler.model.autoencoder['encoder'].add_extra_return = True
                tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device, add_dummy_dim=True))
                train_handler.model.autoencoder['encoder'].add_extra_return = False

                #print(tdict['model'].keys())
                uses_attn = any([f'attn_scores' in k for k in tdict.keys()])
                if not uses_attn:
                    plt.close(fig)
                    dataset.reset_max_day() # very important!!
                    dataset.calcule_precomputed()
                    return

                for kb,b in enumerate(dataset.band_names):
                    lcobjb = lcobj.get_b(b)
                    plot_lightcurve(ax, lcobj, b, label=f'{b} obs', max_day=day)

                    ### attn scores
                    p_onehot = tdict[f'input/onehot.{b}'][...,0] # (n,t)
                    p_attn_scores = tdict[f'model/attn_scores/encz.{b}'][:,SCORES_LAYER] # (n,l,h,qt)>(n,h,qt)
                    assert check_attn_scores(p_attn_scores)
                    p_attn_scores = p_attn_scores.mean(dim=1)[...,None] # (n,h,qt)>(n,qt)>(n,qt,1) # mean attention score among the heads: not a distribution
                    p_attn_scores_min_max = tensor_to_numpy(seq_utils.seq_min_max_norm(p_attn_scores, p_onehot)) # (n,qt,1)
                    # print('p_attn_scores', p_attn_scores[0,:,0])
                    # print('p_attn_scores_min_max', p_attn_scores_min_max[0,:,0])
                    
                    b_len = p_onehot.sum().item()
                    assert b_len<=len(lcobjb), f'{b_len}=={len(lcobjb)}'
                    for i in range(0, b_len):
                        c = COLOR_DICT[b]
                        p = p_attn_scores_min_max[0,i,0]
                        ax.plot(lcobjb.days[i], lcobjb.obs[i], 'o',
                            markersize=MAX_MARKERSIZE*p+MIN_MARKERSIZE*(1-p),
                            markeredgewidth=0,
                            c=c,
                            alpha=alpha,
                            )
                    ax.plot([None], [None], 'o', markeredgewidth=0, c=c, label=f'{b} attention scores', alpha=alpha)

                ### vertical line
                if kday>0:
                    ax.axvline(day, linestyle='--', c='k', label=f'th-day={day:.0f} [days]')

                title = ''
                if k==0:
                    title += f'Multi-band light-curve encoder normalized attention scores'+'\n'
                class_name = dataset.class_names[lcobj.y]
                title += f'{latex_bf_alphabet_count(k)} lcobj={lcobj_names[k]} [{class_name.replace("*", "")}]'+'\n'
                ax.set_title(title[:-1])
                ax.set_ylabel('observation [flux]')
                ax.legend(loc='upper right')
                ax.grid(alpha=0.0)
                xlims[lcobj_name] = ax.get_xlim() if xlims[lcobj_name] is None else xlims[lcobj_name]
                ylims[lcobj_name] = ax.get_ylim() if ylims[lcobj_name] is None else ylims[lcobj_name]
                ax.set_xlim(xlims[lcobj_name])
                ax.set_ylim(ylims[lcobj_name])
                yScalarFormatter = ScalarFormatterClass(useMathText=True)
                yScalarFormatter.set_powerlimits((0, 0))
                ax.yaxis.set_major_formatter(yScalarFormatter)
            
            ax.set_xlabel('observation-time [days]')
            fig.tight_layout()
            plot_animator.append(fig)

    ### save file
    plot_animator.save(reverse=True)
    dataset.reset_max_day() # very important!!
    dataset.calcule_precomputed() # very important!!
    return