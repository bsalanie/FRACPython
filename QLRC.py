""" defines a class for QLRC models """

import numpy as np
import scipy.linalg as spla

from utils import bs_error_abort
from BLP_basic import make_K_BLP



class QLRCModel:
    def __init__(self, f0, f1, neps, K=None, d=None, t=None, make_K=None,
                 f_infty=None, proj_instr=spla.lstsq):
        self.f0, self.f1, self_proj_instr, self.neps = f0, f1, proj_instr, neps
        if K is None:
            self.K = None
            if d is None or t is None or make_K is None:
                bs_error_abort("If K is not given then d, t, and make_K should be.")
            self.d, self.t, self.make_K = d, t, make_K
        else:
            self.K = K
        if f_infty is not None:
            self.f_infty = f_infty
        else:
            self.f_infty = None

    def make_K(self):
        Y = self.Y
        A2 = self.d(Y)
        A33 = self.t(Y, self.neps)
        nproducts = Y['X'].shape[1]
        for j in range(nproducts):
            pass


    def take_data(self, Y, covars='diag'):
        self.Y = Y
        if self.K is None:
            self.K = self.make_K(Y, self.d, self.t, self.neps)


    def estimate_2SLS(self, Y, Z):
        f0_vals = self.f0(Y)
        f1_vals = self.f1(Y)
        K_vals = self.K
        self.f1_Z = self.proj_instr(f1_vals, Z)
        self.K_Z = self.proj_instr(K_vals, Z)
        optimal_Z = np.concatenate((self.f1_Z, self.K_Z), axis=1)
        estimates = spla.lstsq(optimal_Z, f0_vals)
        return estimates


    def simulate(self):
        # if self.f_infty is not None:
        pass