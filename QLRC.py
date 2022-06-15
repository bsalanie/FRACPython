""" defines a class for QLRC models """

import numpy as np
import scipy.linalg as spla
from typing import Callable, Optional

from dataclasses import dataclass, field

from utils import bs_error_abort

def QLRC_check_arguments_(Y, f_1, n_dimensions, n_random_coeffs, instruments):
    if not isinstance(Y, np.ndarray):
        bs_error_abort("Y must be a Numpy array")
    if Y.n_dim != 2:
        bs_error_abort("Y must be a 2-dimensional Numpy array")
    n_observations, n_variables = Y.shape
    if isinstance(f_1, np.ndarray):
        pass
    if not isinstance(instruments, np.ndarray):
        bs_error_abort("Y must be a Numpy array")


class QLRCModel:
    def __init__(self, Y: np.ndarray, A_star: Callable, f_1: Callable | np.ndarray,
                 n_dimensions: int, n_random_coeffs: int,
                 instruments: np.ndarray,
                 f_0: Optional[Callable | np.ndarray]=None,
                 f_infty: Optional[Callable]=None,
                 projection_instruments: Optional[Callable]=spla.lstsq,
                 K: Optional[Callable]=None):
        QLRC_check_arguments_(Y, f_1, n_dimensions, n_random_coeffs, instruments)
        self.A_star, self.n_dimensions, self.n_random_coeffs = A_star, n_dimensions, n_random_coeffs
        self.n_observations = n_observations
        self.f_0, self.f_1, self.instruments, self.projection_instruments, \
            self.K, self.f_infty = f_0, f_1, instruments, projection_instruments, K, f_infty
        if K is None:
            self.K = self.make_K()
        else:
            self.K = K

    def make_K(self):
        def K(data):
            A2 = self.d(data)
            A33 = self.t(data)
            X = data.X
            n_products, n_variables = X.shape
            K_regressors = np.zeros((n_products, n_variables))
            for l in range(n_variables):
                K_regressors[:, l] = spla.solve(A2, A33[:, l]) / 2.0
            return K_regressors
        self.K = K

    def fit(model, data):
        X, S, Z = data.X, data.S, data.Z
        n_markets, n_products, n_variables = X.shape
        if model.n_products != n_products:
            bs_error_abort(f"The model has {model.n_products} products but the data has {n_products}")
        if model.n_markets != n_markets:
            bs_error_abort(f"The model has {model.n_markets} markets but the data has {n_markets}")
        if model.n_eps != n_variables:
            bs_error_abort(
                f"The model has {model.n_eps} random coefficients but the data has {n_variables} variables")
        f0, f1, K = model.f0, model.f1, model.K
        n_TJ = n_markets * n_products
        f0_vals = np.zeros(n_TJ)
        f1_vals = np.zeros((n_TJ, n_variables))
        K_vals = np.zeros((n_TJ, n_variables))
        i = 0
        for mkt in range(n_markets):
            slice_mkt = slice(i, i + n_products)
            data_mkt = data.get_mkt(mkt)
            f0_vals[slice_mkt] = f0(data_mkt)
            f1_vals[slice_mkt, :] = f1(data_mkt)
            K_vals[slice_mkt, :] = K(data_mkt)
        n_instruments = Z.shape[2]
        Z_TJ = np.zeros((n_TJ, n_instruments))
        for instr in range(n_instruments):
            Z_TJ[:, instr] = Z[:, :, instr].reshape(n_TJ)
        f1_Z = model.proj_instr(f1_vals, Z_TJ)
        K_Z = model.proj_instr(K_vals, Z_TJ)
        optimal_Z = np.concatenate((f1_Z, K_Z), axis=1)
        estimates = spla.lstsq(optimal_Z, f0_vals)
        return estimates

    def predict(self, f_infty):
        pass

    def fit_corrected(self):
        pass

    def print(self):
        pass



# from BLP_basic import make_K_BLP
#
# """
# diagonal case, no micromoments, random coeff on each variable
#
# on each market we need:
#  * a function f0 (S_1...S_J) -> [J]
#  * a function f1 (Y) -> [J,p]
#  * either:
#     a function that returns a set of regressors K[j,p] from the data
#     or d: A2 [J,J]
#        t: A33[J,p]
#        and we solve A2.K = A33/2
#
# then we need the data Y:
#    [T, J] market shares S
#    [T, J, p] covariates X
#    {T, J, m] instruments Z
#
#
# and a function estimate_2SLS that takes both and returns the estimates
#
# """
#
#
#
# @dataclass
# class DataBLPOneMarket:
#     S: np.ndarray
#     X: np.ndarray
#     Z: np.ndarray
#
#
# @dataclass
# class QLRCDataBLP:
#     S: np.ndarray
#     X: np.ndarray
#     Z: np.ndarray
#     n_markets: int = field(init=False)
#     n_products: int = field(init=False)
#     n_variables: int = field(init=False)
#     n_instruments: int = field(init=False)
#
#     def __post_init__(self):
#         if self.S.ndim != 2:
#             bs_error_abort(f"S should be (T,J) not {self.S.shape}")
#         n_markets, n_products = self.S.shape
#         if self.X.ndim != 3:
#             bs_error_abort(f"X should be (T,J,p) not {self.X.shape}")
#         if self.X.shape[:2] != (n_markets, n_products):
#             bs_error_abort(f"X is {self.X.shape} but S is {self.S.shape}")
#         if self.Z.ndim != 3:
#             bs_error_abort(f"Z should be (T,J,m) not {self.Z.shape}")
#         if self.Z.shape[:2] != (n_markets, n_products):
#             bs_error_abort(f"Z is {self.Z.shape} but S is {self.S.shape}")
#         self.n_markets, self.n_products, self.n_variables = self.X.shape
#         self.n_instruments = self.Z.shape[2]
#
#     def get_market(self, mkt):
#         return DataBLPOneMarket(self.S[mkt, :], self.X[mkt, :, :], self.Z[mkt, :, :])
#


