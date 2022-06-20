import numpy as np
from typing import Union, Tuple
from itertools import combinations_with_replacement
from functools import reduce
import scipy.linalg as spla
import sys



from statsmodels.nonparametric._kernel_base import EstimatorSettings
from statsmodels.nonparametric.kernel_regression import KernelReg


def _powers_Z(Z: np.array, list_ints: list) -> np.array:
    """
    used internally by :func:`proj_Z`; returns :math:`\\prod_{k=1}^m  Z_{\\cdot k}^{l_k}`

    :param np.array Z: a matrix `(n, m)`

    :param list list_ints: a list of integers

    :return: the product of the powers of Z
    """
    assert Z.ndim == 2, f"powers_Z_: Z should have dimension 2"
    m = Z.shape[1]
    assert len(list_ints) == m, "The length of list_ints should equal the number of columns of Z"
    i_powers = list(zip(np.arange(m), list_ints))
    return reduce(lambda x, y: x * (Z[:, y[0]] ** y[1]), i_powers, 1)


def proj_Z(W: np.array, Z: np.array, p: int = 1, verbose: bool=False) \
        -> Tuple[np.array, np.array, float]:
    """
    project `W` on `Z` up to degree `p` interactions

    :param  np.array W: variable(s) `(nobs)` or `(nobs, nw)`

    :param  np.array Z: instruments `(nobs) or `(nobs, nz)`;  they should **not** include a constant term

    :param int p: maximum total degree for interactions of the columns of `Z`

    :param bool verbose: prints stuff if True

    :return: projections of the columns of `W` on `Z` etc, coefficients, and :math:`R^2` of each column
    """

    nobs = Z.shape[0]
    assert W.shape[0] == nobs, "proj_Z: W and Z should have the same number of rows"
    assert W.ndim <= 2, "proj_Z: W should have 1 or 2 dimensions"
    assert Z.ndim <= 2, "proj_Z: Z should have 1 or 2 dimensions"

    if Z.ndim == 1:
        Zp = np.zeros((nobs, 1 + p))
        Zp[:, 0] = np.ones(nobs)
        for q in range(1, p + 1):
            Zp[:, q] = Z ** q
    else:  # Z is a matrix
        m = Z.shape[1]
        list_vars = list(range(m))
        MAX_NTERMS = round(nobs / 5)
        Zp = np.zeros((nobs, MAX_NTERMS))
        Zp[:, 0] = np.ones(nobs)
        k = 1
        for q in range(1, p + 1):
            listq = list(combinations_with_replacement(list_vars, q))
            lenq = len(listq)
            l = np.zeros((m, lenq))
            for i in range(m):
                l[i, :] = np.array(list(map(lambda x: x.count(i), listq)))
            for j in range(lenq):
                Zp[:, k] = _powers_Z(Z, l[:, j])
                k += 1
                assert k < MAX_NTERMS, f"proj_Z: we don't allow more than {MAX_NTERMS} terms"
        Zp = Zp[:, :k]
        if verbose:
            print(f"_proj_Z with degree {p}, using {k} regressors")

    MINVAR = 1e-12
    b_proj, _, _, _ = spla.lstsq(Zp, W)
    W_proj = Zp @ b_proj
    r2 = 1.0
    if W.ndim == 1:
        var_w = np.var(W)
        if var_w > MINVAR:
            r2 = np.var(W_proj) / var_w
    elif W.ndim == 2:
        nw = W.shape[1]
        r2 = np.ones(nw)
        for i in range(nw):
            var_w = np.var(W[:, i])
            if var_w > MINVAR:
                r2[i] = np.var(W_proj[:, i]) / var_w
    else:
        sys.exit("proj_Z: Wrong number of dimensions {W.ndim} for W")
    return W_proj, b_proj, r2



def reg_nonpar(y: np.array, X: np.array, var_types: str = None,
               n_sub: int = None, n_res: int = 1):
    """
    nonparametric regression of y on the columns of X;
    bandwidth chosen on a subsample of size nsub if nsub < nobs, and rescaled

    :param np.array y: a vector of size nobs

    :param np.array X: a (nobs) vector or a matrix of shape (nobs, m)

    :param str var_types: specify types of all `X` variables if not all of them are continuous; \
     one character per variable

      * 'c' for continuous
      * 'u' discrete unordered
      * 'o' discrete ordered

    :param n_sub: size of subsample for cross-validation;  by default it is :math:`200^{(m+4)/5}`

    :param int n_res: how many subsamples we draw; 1 by default

    :return: fitted on sample (nobs, with derivatives)  and bandwidths (m)
    """
    assert X.ndim == 1 or X.ndim == 2, "X should be a vector or a matrix"
    assert y.ndim == 1, "y should be a vector"
    n_obs = y.size
    assert X.shape[0] == n_obs, "X and y should have the same number of observations"
    m = 1 if X.ndim == 1 else X.shape[1]
    if var_types is None:
        types = 'c' * m
    else:
        assert len(var_types) == m, \
            "var_types should have one entry for each column of X"
        types = var_types

    if n_sub is None:
        n_sub = round(200**((m+4.0)/5.0))

    k = KernelReg(y, X, var_type=types,
                  defaults=EstimatorSettings(efficient=True, n_sub=n_sub,
                                             randomize=True, n_res=n_res))
    return k.fit(), k.bw


def reg_nonpar_fit(y: np.array, X: np.array, var_types: str = None,
                   n_sub: int = None, n_res: int=1,
                   verbose: bool=False) -> np.array:
    """
    nonparametric regression of y on the columns of X; bandwidth chosen on a subsample of size nsub if nsub < nobs, and rescaled

    :param np.array y: a vector of size nobs

    :param np.array X: a (nobs) vector or a matrix of shape (nobs, m)

    :param str var_types: specify types of all `X` variables if not all of them are continuous; \
     one character per variable

      * 'c' for continuous
      * 'u' discrete unordered
      * 'o' discrete ordered

    :param n_sub: size of subsample for cross-validation; by default it is :math:`200^{(m+4)/5}`

    :param int n_res: how many subsamples we draw; 1 by default

    :param bool verbose: prints stuff if True

    :return: fitted on sample (nobs)
    """
    kfbw = reg_nonpar(y, X, var_types, n_sub, n_res)
    return kfbw[0][0]



def flexible_reg(Y: np.array, X: np.array, mode: str = 'NP',
                 var_types: str = None, n_sub: int = None,
                 n_res: int = 1, verbose: bool=False):
    """
    flexible regression  of `Y` on `X`

    :param np.array Y: independent variable `(nobs)` or `(nobs, ny)`

    :param np.array X: covariates `(nobs)` or `(nobs, nx)`; should **not** include a constant term

    :param str mode: what flexible means

    * 'NP': non parametric
    * '1': linear
    * '2': quadratic, etc

    :param str var_types: [for 'NP' only]  specify types of all `X` variables if not all of them are continuous; \
     one character per variable

     * 'c' for continuous
     * 'u' discrete unordered
     * 'o' discrete ordered

    :param int n_sub: [for 'NP' only] size of subsample for cross-validation; \
    by default it is :math:`200^{(m+4)/5}`

    :param int n_res: [for 'NP' only] how many subsamples we draw; 1 if `None`

    :param bool verbose: prints stuff if True

    :return: :math:`E(y|X)` at the sample points
    """
    if mode == 'NP':
        if Y.ndim == 2:
            ny = Y.shape[1]
            Y_fit = np.zeros_like(Y)
            for iy in range(ny):
                Y_fit[:, iy] = reg_nonpar_fit(Y[:, iy], X, var_types=var_types,
                                              n_sub=n_sub, n_res=n_res, verbose=verbose)
            return Y_fit
        else:
            return reg_nonpar_fit(Y, X, var_types=var_types, n_sub=n_sub, n_res=n_res, verbose=verbose)
    else:
        try:
            imode = int(mode)
        except TypeError:
            print(f"flexible_reg does not accept mode={mode}")
            sys.exit(1)
        preg, _, _ = proj_Z(Y, X, p=imode, verbose=verbose)
        return preg