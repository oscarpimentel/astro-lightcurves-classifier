from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer

EPS = 1e-5
LOG_EPS = 1
USES_MEAN = True

###################################################################################################################################################

class CustomStandardScaler():
    def __init__(self,
        uses_mean=USES_MEAN,
        eps=EPS,
        ):
        self.uses_mean = uses_mean
        self.eps = eps
        self.reset()
        
    def reset(self):
        pass

    def fit(self, x):
        assert len(x.shape)==2 # (n,f)
        self.m = x.mean(axis=0)[None].astype(x.dtype) if self.uses_mean else 0.
        self.s = x.std(axis=0)[None].astype(x.dtype)

    def transform(self, x):
        assert len(x.shape)==2 # (n,f)
        z = (x-self.m)/(self.s+self.eps)
        return z

    def inverse_transform(self, z):
        assert len(z.shape)==2 # (n,f)
        x = z*(self.s+self.eps)+self.m
        return x

###################################################################################################################################################

class LogStandardScaler():
    def __init__(self,
        eps=LOG_EPS,
        uses_mean=USES_MEAN,
        ):
        self.eps = eps
        self.uses_mean = uses_mean
        self.reset()

    def reset(self):
        self.scaler = CustomStandardScaler(
            uses_mean=self.uses_mean,
            )

    def fit(self, x):
        assert np.all(x>=0)
        log_x = np.log(x+self.eps)
        self.scaler.fit(log_x)
        
    def transform(self, x):
        assert np.all(x>=0)
        log_x = np.log(x+self.eps)
        z = self.scaler.transform(log_x)
        return z
    
    def inverse_transform(self, z):
        log_x = self.scaler.inverse_transform(z) # np.log(x+self.eps)
        x = np.exp(log_x)-self.eps
        return x