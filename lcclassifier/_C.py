import os
import lchandler._C as _Clchandler


# settings
ATTN_DROPOUT = 0 / 100.
EPS = 1e-5
RANDOM_STATE = 0
BASELINE_D = {
    'mdl=BRF~fmode=all~training-set=[r]': {
        'path': os.path.join('data', 'method=spm-mcmc-estw~tmode=r~fmode=all/performance/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method=spm-mcmc-estw'),
    },
    'mdl=BRF~fmode=all~training-set=spm-mcmc-estw[s]': {
        'path': os.path.join('data', 'method=spm-mcmc-estw~tmode=s~fmode=all/performance/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method=spm-mcmc-estw'),
    },
    # 'r+s': {
    #     'path': 'data/method=spm-mcmc-estw~tmode=r+s~fmode=all/performance/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method=spm-mcmc-estw',
    #     'label': 'BRF (fmode=all; training-set=[r+s])',
    # },
}
METRICS_D = {
    'aucroc': {'k': 1, 'mn': 'AUCROC'},
    'precision': {'k': 1, 'mn': 'Precision'},
    'recall': {'k': 1, 'mn': 'Recall'},
    'f1score': {'k': 1, 'mn': '$F_1$score'},
    'aucpr': {'k': 1, 'mn': 'AUCPR'},
    'gmean': {'k': 1, 'mn': 'Gmean'},
    'accuracy': {'k': 1, 'mn': 'Accuracy'},
}

# loss
REC_LOSS_EPS = 1
REC_LOSS_K = 1e-3  # 10 1e-3 0
WMSE_K = 10000  # 0 1000 5000
XENTROPY_K = 1

DEFAULT_MIN_DAY = 1
MAX_DAY = 100
DEFAULT_DAYS_N = int(MAX_DAY - DEFAULT_MIN_DAY + 1)
DEFAULT_DAYS_N_AN = 2  # 2 50

# dicts
CLASSES_STYLES = _Clchandler.CLASSES_STYLES
COLOR_DICT = _Clchandler.COLOR_DICT

# patches
KEYS_TO_CHANGE_D = {'.te_film': '.time_film'}  # patch to revive old state_dicts after changes in repo
