from __future__ import print_function
from __future__ import division
from . import _C

import fuzzytools.strings as strings
import fuzzytools.matplotlib.colors as ftc
import numpy as np
import fuzzytools.files as ftfiles

INVALID_MODEL_KEYS = [
    'dummy_seft',
    'enc_emb',
    'dec_emb',
    'input_dims',
    'kernel_size',
    'b',

    'pb',
    'bypass_synth',
    'bypass_prob',
    'ds_prob',
]
MODEL_KEYS_REPLACE = {
    'time_noise_window': '$\\varepsilon_t$',
    'heads': 'H',
}


def get_model_names(rootdir, cfilename, kf, lcset_name,
    train_mode='fine-tuning',
    ):
    roodirs = [r.split('/')[-1] for r in ftfiles.get_roodirs(rootdir)]
    model_names = [r for r in roodirs if '~' in r]
    model_names.sort()
    return model_names


def get_fmodel_name(model_name,
    returns_mn_dict=False,
    ):
    mn_dict = strings.get_dict_from_string(model_name)
    mdl_info = []
    for k in mn_dict.keys():
        v = mn_dict[k]
        if v is None or v=='None':
            continue
        if k in INVALID_MODEL_KEYS+['mdl']:
            continue
        if k=='m': # patch
            txt = f'M={int(v)//2}'
        else:
            txt = f'{MODEL_KEYS_REPLACE.get(k, k)}={v}'
        mdl_info += [txt]

    mdl_name = mn_dict['mdl']
    mdl_info = '; '.join(mdl_info)
    fmodel_name = f'{mdl_name} ({mdl_info})'
    fmodel_name = fmodel_name.replace('RNN', 'RNN+$\\Delta t$')
    fmodel_name = fmodel_name.replace('*24**-1', '/24')
    fmodel_name = fmodel_name.replace('Parallel', 'P-')
    fmodel_name = fmodel_name.replace('Serial', 'S-')
    if returns_mn_dict:
        return fmodel_name, mn_dict
    else:
        return fmodel_name


def get_sorted_model_names(model_names,
                           merged=True,
                           ):
    p_model_names = []
    s_model_names = []
    for model_name in model_names:
        is_parallel = 'Parallel' in model_name
        if is_parallel:
            p_model_names += [model_name]
        else:
            s_model_names += [model_name]
    p_model_names = sorted(p_model_names)
    s_model_names = sorted(s_model_names)
    if merged:
        return s_model_names + p_model_names
    else:
        return s_model_names, p_model_names


def get_cmetric_name(metric_name):
    #metric_name = metric_name.replace('accuracy', 'acc')
    #metric_name = metric_name.replace('f1score', 'f1s')
    return metric_name

def get_mday_str(metric_name, day_to_metric):
    new_metric_name = get_cmetric_name(metric_name)
    return new_metric_name+'$|^{'+str(day_to_metric)+'}$'

def get_mday_avg_str(metric_name, day_to_metric,
    first_day=2,
    ):
    new_metric_name = get_cmetric_name(metric_name)
    return new_metric_name+'$|_{'+str(first_day)+'}'+'^{'+str(day_to_metric)+'}$'
    
###################################################################################################################################################

def filter_models(model_names, condition_dict):
    new_model_names = []
    for model_name in model_names:
        mn_dict = strings.get_dict_from_string(model_name)
        conds = []
        for c in condition_dict.keys():
            value = mn_dict.get(c, None)
            acceptable_values = condition_dict[c]
            conds += [value in acceptable_values]
        if all(conds):
            new_model_names += [model_name]
    return new_model_names

def _get_unique_model_name(model_name):
    _, mn_dict = get_fmodel_name(model_name, returns_mn_dict=True)
    mn_dict.pop('heads', None)
    unique_model_name = get_fmodel_name(strings.get_string_from_dict(mn_dict))
    unique_model_name = unique_model_name.replace('S-', '').replace('P-', '')
    return unique_model_name

def get_model_color(target_model_name, model_names):
    unique_model_names = []
    for model_name in model_names:
        unique_model_name = _get_unique_model_name(model_name)
        if not unique_model_name in unique_model_names:
            unique_model_names += [unique_model_name]

    color_dict = ftc.get_color_dict(unique_model_names)
    color = color_dict[_get_unique_model_name(target_model_name)]
    return color