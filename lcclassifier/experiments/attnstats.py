from __future__ import print_function
from __future__ import division
from . import _C

import torch
from fuzzytorch.utils import TDictHolder, tensor_to_numpy, minibatch_dict_collate
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
import fuzzytools.datascience.metrics as fcm
from fuzzytools.matplotlib.utils import save_fig
import matplotlib.pyplot as plt
import fuzzytorch.models.seq_utils as seq_utils
from scipy.optimize import curve_fit
from lchandler import _C as _Clchandler
from lchandler.plots.lc import plot_lightcurve
from .utils import check_attn_scores

SCORES_LAYER = -1
EPS = _C.EPS


def local_slope_f(time, m, n):
    return time*m+n


def get_local_slope(days, obs, j, dj,
    p0=[0,0],
    ):
    assert not dj%2==0
    assert dj>=3
    ji = max(0, j-dj//2)
    jf = min(j+dj//2+1, len(obs))
    sub_days = days[ji:jf] # sequence steps
    sub_obs = obs[ji:jf] # sequence steps
    popt, pcov = curve_fit(local_slope_f, sub_days, sub_obs, p0=p0)
    local_slope_m, local_slope_n = popt
    return local_slope_m, local_slope_n, sub_days, sub_obs


def save_attnstats(train_handler, data_loader, save_rootdir,
    eps:float=EPS,
    dj=3,
    min_len=3,
    **kwargs):
    train_handler.load_model(keys_to_change_d=_C.KEYS_TO_CHANGE_D) # important, refresh to best model
    train_handler.model.eval() # important, model eval mode
    dataset = data_loader.dataset # get dataset

    is_parallel = 'Parallel' in train_handler.model.get_name()
    if not is_parallel:
        return

    attn_scores_collection = {b:[] for kb,b in enumerate(dataset.band_names)}
    with torch.no_grad():
        tdicts = []
        for ki,in_tdict in enumerate(data_loader):
            train_handler.model.autoencoder['encoder'].add_extra_return = True
            _tdict = train_handler.model(TDictHolder(in_tdict).to(train_handler.device))
            train_handler.model.autoencoder['encoder'].add_extra_return = False
            tdicts += [_tdict]
        tdict = minibatch_dict_collate(tdicts)

        for kb,b in enumerate(dataset.band_names):
            p_onehot = tdict[f'input/onehot.{b}'][...,0] # (n,t)
            #p_rtime = tdict[f'input/rtime.{b}'][...,0] # (n,t)
            #p_dtime = tdict[f'input/dtime.{b}'][...,0] # (n,t)
            #p_x = tdict[f'input/x.{b}'] # (n,t,i)
            #p_rerror = tdict[f'target/rerror.{b}'] # (n,t,1)
            #p_rx = tdict[f'target/recx.{b}'] # (n,t,1)

            # print(tdict.keys())
            uses_attn = any([f'attn_scores' in k for k in tdict.keys()])
            if not uses_attn:
                return

            ### attn scores
            p_attn_scores = tdict[f'model/attn_scores/encz.{b}'][:,SCORES_LAYER] # (n,l,h,qt)>(n,h,qt)
            assert check_attn_scores(p_attn_scores)
            p_attn_scores_mean = p_attn_scores.mean(dim=1)[...,None] # (n,h,qt)>(n,qt)>(n,qt,1) # mean attention score among the heads: not a distribution
            p_attn_scores_min_max = seq_utils.seq_min_max_norm(p_attn_scores_mean, p_onehot) # (n,qt,1)

            ### stats
            lcobj_names = dataset.get_lcobj_names()
            bar = ProgressBar(len(lcobj_names))
            for k,lcobj_name in enumerate(lcobj_names):
                lcobj = dataset.lcset[lcobj_name]
                lcobjb = lcobj.get_b(b) # complete lc
                p_onehot_k = tensor_to_numpy(p_onehot[k]) # (n,t)>(t)
                b_len = p_onehot_k.sum()
                assert b_len<=len(lcobjb), f'{b_len}<={len(lcobjb)}'

                if not b_len>=min_len:
                    continue

                p_attn_scores_k = tensor_to_numpy(p_attn_scores_mean[k,:b_len,0]) # (n,qt,1)>(t)
                p_attn_scores_min_max_k = tensor_to_numpy(p_attn_scores_min_max[k,:b_len,0]) # (n,qt,1)>(t)

                days = lcobjb.days[:b_len] # (t)
                obs = lcobjb.obs[:b_len] # (t)
                obse = lcobjb.obse[:b_len] # (t)
                snr = lcobjb.get_snr(max_len=b_len)
                max_obs = np.max(obs)
                peak_day = days[np.argmax(obs)]
                duration = days[-1]-days[0]

                bar(f'b={b}; lcobj_name={lcobj_name}; b_len={b_len}; snr={snr}; max_obs={max_obs}')
                lc_features = []
                for j in range(0, b_len):
                    j_features = {
                        f'j':j,
                        f'attn_scores_k.j':p_attn_scores_k[j],
                        f'attn_scores_min_max_k.j':p_attn_scores_min_max_k[j],
                        f'days.j':days[j],
                        f'obs.j':obs[j],
                        f'obse.j':obse[j],
                        }
                    local_slope_m, local_slope_n, sub_days, sub_obs = get_local_slope(days, obs, j, dj) # get local slope
                    j_features.update({
                        f'local_slope_m.j~dj={dj}':local_slope_m,
                        f'local_slope_n.j~dj={dj}':local_slope_n,
                        f'peak_distance.j~dj={dj}~mode=local':days[j]-peak_day,
                        f'peak_distance.j~dj={dj}~mode=mean':np.mean(sub_days)-peak_day,
                        f'peak_distance.j~dj={dj}~mode=median':np.median(sub_days)-peak_day,
                        })
                    lc_features += [j_features]

                attn_scores_collection[b] += [{
                    f'c':dataset.class_names[lcobj.y],
                    f'b_len':b_len,
                    f'peak_day':peak_day,
                    f'duration':duration,
                    f'snr':snr,
                    f'max_obs':max_obs,
                    f'lc_features':lc_features,
                    }]
    bar.done()
    results = {
        'model_name':train_handler.model.get_name(),
        'survey':dataset.survey,
        'band_names':dataset.band_names,
        'class_names':dataset.class_names,

        'max_day':dataset.max_day,
        'attn_scores_collection':attn_scores_collection,
    }

    ### save file
    save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}.d'
    files.save_pickle(save_filedir, results) # save file
    dataset.reset_max_day() # very important!!
    dataset.calcule_precomputed() # very important!!
    return