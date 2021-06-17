import numpy as np
import numpy.linalg as npla
from typing import Tuple
import sys

from utils import print_stars, bs_error_abort, npmaxabs


def f0_BLP(Y):
    S = Y['shares']
    S0 = 1.0 - np.sum(S, 1)
    return np.log(S / S0.reshape((-1, 1)))


def f1_BLP(Y):
    return Y['X']


def d_BLP(Y):
    S = Y['shares']
    nmarkets, nproducts = S.shape
    d = np.zeros((nmarkets, nproducts, nproducts))
    for t, St in enumerate(S):
        d[t, :, :] = np.diag(St) - np.outer(St, St)
    return d


def t_BLP(Y, neps):
    S = Y['shares']
    nmarkets, nproducts = S.shape
    A33 = np.zeros((nmarkets, nproducts, neps, neps))
    X = Y['X']
    for t in range(nmarkets):
        St, Xt = S[t, :], X[t, :, :]
        eS_Xt = np.sum(Xt* St.reshape((-1,1)), 1)
        Xthat = Xt - eS_Xt
        eS_XXt = np.zeros((neps, neps))
        for ieps in range(neps):
            XteS = Xt[:, ieps]*St
            eS_XXt[ieps, :] = np.sum(Xt* XteS.reshape((-1,1)), 1)
        At = np.outer(eS_Xt, eS_Xt) - eS_XXt
        for j in range(nproducts):
            Xthat_j = Xthat[j, :]
            A33[t, j, :, :] = St[j]*(np.outer(Xthat_j, Xthat_j) - At)
    return A33


def make_K_BLP(X: np.array, shares: np.array, covars: str = 'diag') -> np.array:
    """
    for Salanie-Wolak TSLS: 2nd order artificial regressors for a vector or matrix `X` on one market

    :param np.array X: array `(nproducts)` or `(nproducts, nx)`

    :param np.array shares: array `(nproducts)`

    :param str covars: restrictions on variance-covariance of random coefficients

      * 'diag': diagonal only
      * 'all': all terms

    :return: the  `K` regressors, an array `(nproducts)` or  `(nproducts, nx)` or `(nproducts, nx*(nx+1)/2)`
    """
    if X.ndim == 1:
        assert X.size == shares.size, "if X is a vector, X and shares should have the same size"
        eS_X = np.dot(shares, X)
        djm = eS_X - X / 2.0
        return -djm * X
    elif X.ndim == 2:
        assert X.shape[0] == shares.size, "if X is a matrix, X and shares should have the same number of rows"
        eS_X = X.T @ shares
        djm = X / 2.0 - eS_X
        if covars == 'diag':
            return djm * X
        elif covars == 'all':
            nproducts, nx = X.shape
            nx12 = (nx * (nx + 1)) / 2
            K = np.array((nproducts, nx12))
            i = 0
            for ix in range(nx):
                K[:, i] = djm[:, ix] * X[:, ix]
                i += 1
                for ix2 in range(ix + 1, nx):
                    K[:, i] = djm[:, ix] * X[:, ix2] + djm[:, ix2] * X[:, ix]
                    i += 1
            return K
        else:
            bs_error_abort(f"covars cannot be {covars}!")
    else:
        bs_error_abort("X must be a vector or matrix")


def K_BLP_diag(Y):
    return make_K_BLP(Y['X'], Y['shares'], covars='diag')


def f_infty_BLP(Y, Sigma):
    pass




def simulated_shares(utils: np.array) -> np.array:
    """
    return simulated shares for given simulated utilities

    :param np.array utils: array `(nproducts, ndraws)`

    :return: simulated shares `(nproducts, ndraws)`
    """
    shares = np.exp(utils)
    denom = 1.0 + np.sum(shares, 0)
    shares = shares / denom
    return shares


def simulated_mean_shares(utils: np.array) -> np.array:
    """
    return simulated mean shares for given simulated utilities

    :param np.array utils: array `(nproducts, ndraws)`

    :return: np.array simulated mean shares: array `(nproducts)`
    """
    return np.mean(simulated_shares(utils), 1)


def berry_core(shares: np.array, mean_u: np.array, X2: np.array,
               Sigma: np.array, tol: float = 1e-9,
               maxiter: int = 10000, ndraws: int = 10000, verbose: bool = False) -> Tuple[np.array, bool]:
    """
    contraction to invert for product effects :math:`\\xi` from market shares

    :param np.array shares: `nproducts` vector of observed market shares

    :param np.array mean_u: `(nproducts)` vector of mean utilities

    :param np.array X2: `(nproducts, nx2)` matrix of nonlinear covariates

    :param np.array Sigma: `(nx2, nx2)` variance-covariance matrix of random coefficients on `X2`, \
    or `(nx2)` if diagonal

    :param float tol: tolerance

    :param int maxiter: max iterations

    :param int ndraws: number of draws for simulation

    :params bool verbose: print stuff if `True`

    :return: `(nproducts)` vector of :math:`\\xi` values, and return code 0 if OK
    """
    nproducts, nx2 = X2.shape
    assert shares.size == nproducts, "should have as many shares as rows in X2"
    assert mean_u.size == nproducts, "should have as many mean utilities as rows in X2"

    if Sigma.ndim == 1 and Sigma.size == nx2:
        assert np.min(Sigma) >= 0.0, "berry_core: all elements of the diagonal Sigma should be positive or 0"
        Xsig = X2 * np.sqrt(Sigma)
    elif Sigma.ndim == 2 and Sigma.shape == (nx2, nx2):
        L = npla.cholesky(Sigma)
        Xsig = X2 @ L
    else:
        print_stars("berry_core: Sigma should be (nx2, nx2) or (nx2)")
        sys.exit()

    sum_shares = shares.sum()
    market_zero_share = 1.0 - sum_shares

    xi = np.log(shares / market_zero_share) - mean_u
    max_err = np.Inf
    retcode = 0
    iter = 0
    eps = np.random.normal(size=(nx2, ndraws))
    while max_err > tol:
        utils = (Xsig @ eps) + (mean_u + xi).reshape((-1, 1))
        shares_sim = simulated_mean_shares(utils)
        err_shares = shares - shares_sim
        max_err = npmaxabs(err_shares)
        if verbose and iter % 100 == 0:
            print(f"berry_core: error {max_err} after {iter} iterations")
        iter += 1
        if iter > maxiter:
            retcode = 1
            break
        xi += (np.log(shares) - np.log(shares_sim))
    if verbose:
        print_stars(f"berry_core: error {max_err} after {iter} iterations")
    return xi, retcode


def berry_normal_diagonal(shares: np.array, X1: np.array, X2: np.array, beta: np.array,
                          sigmas: np.array, tol: float = 1e-9,
                          maxiter: int = 10000, ndraws: int = 10000) -> Tuple[np.array, bool]:
    """
    contraction to invert for product effects :math:`\\xi` from market shares in a normal model \
    with a diagonal variance-covariance matrix and no micromoments

    :param np.array shares: `nproducts` vector of observed market shares

    :param np.array X1: `(nproducts, nx1)` matrix of linear covariates

    :param np.array X2: `(nproducts, nx2)` matrix of nonlinear covariates

    :param np.array beta: `(nx1)` vector of mean coefficients on `X1`

    :param np.array sigmas: `(nx2)` vector of standard errors of random coefficients on `X2`

    :param float tol: tolerance

    :param int maxiter: max iterations

    :param int ndraws: number of draws for simulation

    :return: `(nproducts)` vector of :math:`\\xi` values, and return code 0 if OK
    """
    nx1 = X1.shape[1]
    assert beta.size == nx1, f"berry_inversion:  beta should {nx1} elements"
    mean_utils = X1 @ beta
    Sigma = np.diag(sigmas * sigmas)
    return berry_core(shares, mean_utils, X2, Sigma, tol=tol,
                      maxiter=maxiter, ndraws=ndraws)
