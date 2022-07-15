from __future__ import print_function
from __future__ import division
from . import _C

import torch
from fuzzytorch.utils import get_model_name, TDictHolder, minibatch_dict_collate
import numpy as np
from fuzzytools.progress_bars import ProgressBar, ProgressBarMulti
import fuzzytools.files as files
import fuzzytools.datascience.metrics as fcm
from fuzzytools.matplotlib.utils import save_fig
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts
import matplotlib.pyplot as plt
import fuzzytorch.models.seq_utils as seq_utils
import pandas as pd

DEFAULT_DAYS_N = _C.DEFAULT_DAYS_N
DEFAULT_MIN_DAY = _C.DEFAULT_MIN_DAY

###################################################################################################################################################

def save_temporal_modulation(train_handler, data_loader, save_rootdir,
    days_n:int=DEFAULT_DAYS_N,
    **kwargs):
    train_handler.load_model(keys_to_change_d=_C.KEYS_TO_CHANGE_D) # important, refresh to best model
    train_handler.model.eval() # model eval
    dataset = data_loader.dataset # get dataset

    if not hasattr(train_handler.model, 'get_info'):
        return

    days = np.linspace(DEFAULT_MIN_DAY, dataset.max_day, days_n)#[::-1]
    temporal_encoding_info = train_handler.model.get_info()
    #print('temporal_encoding_info',temporal_encoding_info)

    results = {
        'model_name':train_handler.model.get_name(),
        'survey':dataset.survey,
        'band_names':dataset.band_names,
        'class_names':dataset.class_names,

        'days':days,
        'temporal_encoding_info':temporal_encoding_info,
    }

    ### save file
    save_filedir = f'{save_rootdir}/{dataset.lcset_name}/id={train_handler.id}.d'
    files.save_pickle(save_filedir, results) # save file
    return