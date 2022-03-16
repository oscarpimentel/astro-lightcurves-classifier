import numpy as np
import lchandler._C as _Clchandler

###################################################################################################################################################

ATTN_DROPOUT = 0/100
EPS = 1e-5
RANDOM_STATE = 0
BASELINE_ROOTDIR = '../data/method=spm-mcmc-estw~tmode=r~fmode=all/performance/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method=spm-mcmc-estw'
BASELINE_MODEL_NAME = f'mdl=BRF~fmode=all~training-set=[r]'
METRICS_D = {
	f'aucroc':{'k':1, 'mn':'AUCROC'},
	f'precision':{'k':1, 'mn':'Precision'},
	f'recall':{'k':1, 'mn':'Recall'},
	f'f1score':{'k':1, 'mn':'$F_1$score'},
	f'aucpr':{'k':1, 'mn':'AUCPR'},
	f'gmean':{'k':1, 'mn':'Gmean'},
	f'accuracy':{'k':1, 'mn':'Accuracy'},
	}

### loss
REC_LOSS_EPS = 1
REC_LOSS_K = 1e-3 # 10 1e-3 0
MSE_K = 10000 # 0 1000 5000
XENTROPY_K = 1

DEFAULT_MIN_DAY = 1
MAX_DAY = 100
DEFAULT_DAYS_N = int(MAX_DAY-DEFAULT_MIN_DAY+1)
DEFAULT_DAYS_N_AN = 50 # 2 50

### DICTS
CLASSES_STYLES = _Clchandler.CLASSES_STYLES
COLOR_DICT = _Clchandler.COLOR_DICT