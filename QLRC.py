""" defines a class for QLRC models """

import numpy as np
import scipy.linalg as spla
from typing import Callable, Optional

from bsutils import bs_error_abort
from bsnputils import test_vector, test_matrix, test_tensor


def QLRC_check_YZ_(Y, Z):
    n_observations, n_equations, n_Y = test_tensor(Y, 3, "QLRCModel init")
    no_z, ne_z, n_Z = test_tensor(Z, 3, "QLRCModel init")
    if no_z != n_observations:
        bs_error_abort(f"Z has {no_z} elements in its first dimension  but {n_observations=}")
    if ne_z != n_equations:
        bs_error_abort(f"Z has {ne_z} elements in its second dimension  but {n_equations=}")
    return Y, Z, n_observations, n_equations, n_Y, n_Z


def check_K_(K, n_observations, n_equations, n_Sigma):
    no_K, ne_K, nr_K = test_tensor(K, 3, "QLRCModel init")
    if no_K != n_observations:
        bs_error_abort(f"K has {no_K} elements in its first dimension but {n_observations=}")
    if ne_K != n_equations:
        bs_error_abort(f"K has {ne_K} elements in its second dimension but {n_equations=}")
    if nr_K != n_Sigma:
        bs_error_abort(f"K has {ne_K} elements in its second dimension but {n_Sigma=}")
    return K


def check_f_0_(f_0, n_observations, n_equations):
    nr_0, nc_0 = test_matrix(f_0, "QLRCModel init")
    if nr_0 != n_observations:
        bs_error_abort(f"f_0 has {nr_0} rows but {n_observations=}")
    if nc_0 != n_equations:
        bs_error_abort(f"f_0 has {nc_0} columns but {n_equations=}")
    return f_0


def check_f_1_(f_1, n_observations, n_equations):
    no_1, ne_1, nb_1 = test_tensor(f_1, 3, "QLRCModel init")
    if no_1 != n_observations:
        bs_error_abort(f"f_1 has {no_1} elements in its first dimension but {n_observations=}")
    if ne_1 != n_equations:
        bs_error_abort(f"f_0 has {ne_1} elements in its second dimension but {n_equations=}")
    return f_1, nb_1


def least_squares_proj(Z: np.ndarray, y: np.ndarray):
    coeffs, _, _, _ = spla.lstsq(Z, y)
    return Z @ coeffs


class QLRCModel:
    def __init__(self, Y: np.ndarray, A_star: Callable,
                 f_1: Callable | np.ndarray,
                 n_betas: int,
                 n_Sigma: int,
                 Z: np.ndarray,
                 f_0: Optional[Callable | np.ndarray] = None,
                 f_infty: Optional[Callable] = None,
                 projection_instruments: Optional[Callable] = least_squares_proj,
                 K: Optional[Callable | np.ndarray] = None,
                 args: Optional[list] = None):
        self.Y, self.Z, self.n_observations, self.n_equations, self.n_Y, self.n_Z \
            = QLRC_check_YZ_(Y, Z)
        self.n_betas, self.n_Sigma = n_betas, n_Sigma
        self.A_star = A_star
        self.projection_instruments, self.f_infty \
            = projection_instruments, f_infty
        if f_0 is None:
            pass  # TODO: solve for values of f_0
        elif isinstance(f_0, np.ndarray):
            self.f_0 = check_f_0_(f_0, self.n_observations, self.n_equations)
        elif isinstance(f_0, Callable):
            n_obs = self.n_observations
            f0_test = f_0(Y[0, :, :], args)
            ne_0 = test_vector(f0_test, "QLRCModel init")
            if ne_0 != self.n_equations:
                bs_error_abort(
                    f"f_1 should return a vector of {self.equations} elements, not {ne_0}")
            f0_vals = np.zeros((n_obs, self.n_equations))
            for t in range(n_obs):
                f0_vals[t, :] = f_0(Y[t, :, :], args)
            self.f_0 = f0_vals
        else:
            bs_error_abort("f_0 should be an array, a Callable, or None.")
        if isinstance(f_1, np.ndarray):
            self.f_1 = check_f_1_(f_1, self.n_observations, self.n_equations,
                                  self.n_betas)
        elif isinstance(f_1, Callable):
            n_obs = self.n_observations
            f1_test = f_1(Y[0, :, :], args)
            f1_shape = test_matrix(f1_test, "QLRCModel init")
            if f1_shape != (self.n_equations, self.n_betas):
                bs_error_abort(f"f_1 should return a {(self.n_equations, self.n_betas)} matrix, not a {f1_shape} one")
            f1_vals = np.zeros((n_obs, self.n_equations, n_betas))
            for t in range(n_obs):
                f1_vals[t, :] = f_1(Y[t, :, :], args)
            self.f_1 = f1_vals
        else:
            bs_error_abort("f_1 should be an array or a Callable.")
        self.args = args
        if K is None:
            self.K = self.make_K()
        elif isinstance(K, np.ndarray):
            self.K = check_K_(K, self.n_observations, self.n_equations)
        elif isinstance(K, Callable):
            n_obs = self.n_observations
            K_test = K(Y[0, :, :], args)
            K_shape = test_matrix(K_test, "QLRCModel init")
            if K_shape != (self.n_equations, self.n_Sigma):
                bs_error_abort(f"K should return a {(self.n_equations, self.n_Sigma)} matrix, not a {K_shape} one")
            K_vals = np.zeros((n_obs, self.n_equations, n_Sigma))
            for t in range(n_obs):
                K_vals[t, :] = K(Y[t, :, :], args)
            self.K = K_vals
        else:
            bs_error_abort("K should be an array, a Callable, or None.")
        self.estimated = False

    def make_K(self):
        args, Y = self.args, self.Y
        n_obs, n_eqs, n_Sigma = self.n_observations, self.n_equations, \
                                self.n_Sigma
        A2 = args[1]
        A33 = args[2]
        K = np.zeros((n_obs, n_eqs, n_Sigma))
        for t in range(self.n_observations):
            Y_t = Y[t, :, :]
            A2_vals = A2(Y_t, args)
            A33_vals = A33(Y_t, args)
            K[t, :, :] = spla.solve(A2_vals, A33_vals) / 2.0
            print(f"{K[t,:,:]=}")
        return K

    def fit(self):
        f0_vals, f1_vals, K_vals, Z = self.f_0, self.f_1, self.K, self.Z
        nb, ns, nz = self.n_betas, self.n_Sigma, self.n_Z
        n_points = self.n_observations * self.n_equations
        Zr = np.zeros((n_points, nz))
        for i_z in range(nz):
            Zr[:, i_z] = Z[:, :, i_z].reshape(n_points)
        f0v = f0_vals.reshape(n_points)
        f0r = self.projection_instruments(Zr, f0v)
        f1b = np.zeros((n_points, nb))
        for i_b in range(nb):
            f1b[:, i_b] = f1_vals[:, :, i_b].reshape(n_points)
        f1r = self.projection_instruments(Zr, f1b)
        Ks = np.zeros((n_points, ns))
        for i_s in range(ns):
            Ks[:, i_s] = K_vals[:, :, i_s].reshape(n_points)
        Kr = self.projection_instruments(Zr, Ks)
        optimal_regressors = np.concatenate((f1r, Kr), axis=1)
        estimates, _, _, _ = spla.lstsq(optimal_regressors, f0r)
        self.estimated_betas, self.estimated_Sigma = estimates[:nb], estimates[nb:]
        self.estimated = True
        return estimates

    def predict(self, f_infty):
        pass

    def fit_corrected(self):
        pass

    def print(self):
        print(f"The model has {self.n_observations} observations and {self.n_equations} equations")
        print(f"   there are  {self.n_betas} covariates and {self.n_Sigma} artificial regressors")
        if self.estimated:
            print(f"   the estimates for beta are {self.estimated_betas}")
            print(f"     and those for Sigma are {self.estimated_Sigma}")
