import numpy as np


def simulated_shares_(utils: np.ndarray) -> np.ndarray:
    """
    return simulated shares for given simulated utilities

    :param  utils: array `(nproducts, ndraws)`

    :return: simulated shares `(nproducts, ndraws)`
    """
    shares = np.exp(utils)
    denom = 1.0 + np.sum(shares, 0)
    return shares / denom


def simulated_mean_shares_(utils: np.array) -> np.array:
    """
    return simulated mean shares for given simulated utilities

    :param np.array utils: array `(nproducts, ndraws)`

    :return: np.array simulated mean shares: array `(nproducts)`
    """
    return np.mean(simulated_shares_(utils), 1)


def A_star_BLP(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    computes the `n_products` equations on a given market
    :param a: should be `(n_products,n_Y)`
    :param b: should be `n_products`
    :param c: should be `n_products` (TODO: `n_random_coeffs`)
    :param args: a list `[n_products]`
    :return: a vector `n_products`
    """
    n_products = a.shape[0]
    observed_shares = np.diag(a[:, :n_products])
    shares = np.exp(b + c)
    denom = 1.0 + np.sum(shares)
    return (observed_shares - shares / denom)


def f0_BLP(Y: np.ndarray, args: list):
    """
    computes the `n_products` values of `f_0` on a given market
    :param Y: should be `(n_products,n_Y)`
    :param args: a list `[n_products]`
    :return: a vector `n_products`
    """
    n_products = args[0]
    S = Y[0, :n_products]
    S0 = 1.0 - np.sum(S)
    return np.log(S / S0)


def f1_BLP(Y: np.ndarray, args: list):
    """
    computes the `n_products` values of `f_1` on a given market
    :param Y: should be `(n_products,n_Y)`
    :param args: a list `[n_products]`
    :return: a vector `n_products`
    """
    n_products = args[0]
    X = Y[:, n_products:]
    return X


def A2_BLP(Y: np.ndarray, args: list):
    """
    computes the derivative of :math:`A^\\ast` wrt `b` on a given market
    :param Y: should be `(n_products,n_Y)`
    :param args: a list `[n_products]`
    :return: a matrix `(n_products, n_products)`
    """
    n_products = args[0]
    observed_shares = Y[0, :n_products]
    A_prime_2 = np.diag(observed_shares) - np.outer(observed_shares, observed_shares)
    return A_prime_2


def A33_BLP(Y: np.ndarray, args: list):
    """
    computes the second derivative of :math:`A^\\ast` wrt `c` on a given market
    :param Y: should be `(n_products,n_Y)`
    :param args: a list `[n_products]`
    :return: a matrix `(n_products, n_products)`

    TODO: extend to `n_random_coeffs` and non-diagonal
    """
    n_products = args[0]
    observed_shares = Y[0, :n_products]
    X = Y[:, n_products:]
    eS_X = np.sum(X * observed_shares.reshape((-1, 1)), 0)
    # Xhat = X - eS_X
    XX = X * X
    eS_XX = np.sum(XX * observed_shares.reshape((-1, 1)), 0)
    # A33 = Xhat* Xhat + eS_X*eS_X - eS_XX
    A33 = X * X - eS_XX + 2.0 * (eS_X) * (eS_X) - 2.0 * X * eS_X
    A33 *= observed_shares.reshape((-1, 1))
    # skip non-random constant
    A33 = A33[:, 1:]
    return A33


def K_BLP(Y: np.ndarray, args: list):
    """
    computes the articial regressors on a given market
    :param Y: should be `(n_products,n_Y)`
    :param args: a list `[n_products]`
    :return: a matrix `(n_products, nx-1)`

    TODO: extend to `n_random_coeffs` and non-diagonal
    """
    n_products = args[0]
    observed_shares = Y[0, :n_products]
    X = Y[:, n_products:]
    eS_X = np.sum(X * observed_shares.reshape((-1, 1)), 0)
    K = X * (X / 2.0 - eS_X)
    # non-random constant
    return K[:, 1:]


#
# def make_K_BLP_direct_(X: np.array, S: np.array,
#                covars: str = 'diag') -> np.array:
#     """
#     for Salanie-Wolak TSLS: 2nd order artificial regressors for a vector or matrix `X` on one market
#
#     :param np.array X: array `(nproducts)` or `(nproducts, nx)`
#
#     :param np.array S: array `(nproducts)`
#
#     :param str covars: restrictions on variance-covariance of random coefficients
#
#       * 'diag': diagonal only
#       * 'all': all terms
#
#     :return: the `K` regressors, an array `(nproducts)` or  `(nproducts, nx)` or `(nproducts, nx*(nx+1)/2)`
#     """
#     if X.ndim == 1:
#         assert X.size == S.size, "if X is a vector, X and shares should have the same size"
#         eS_X = np.dot(S, X)
#         djm = eS_X - X / 2.0
#         return -djm * X
#     elif X.ndim == 2:
#         assert X.shape[0] == S.size, "if X is a matrix, X and shares should have the same number of rows"
#         eS_X = X.T @ S
#         djm = X / 2.0 - eS_X
#         if covars == 'diag':
#             return djm * X
#         elif covars == 'all':
#             nproducts, nx = X.shape
#             nx12 = (nx * (nx + 1)) / 2
#             K = np.array((nproducts, nx12))
#             i = 0
#             for ix in range(nx):
#                 K[:, i] = djm[:, ix] * X[:, ix]
#                 i += 1
#                 for ix2 in range(ix + 1, nx):
#                     K[:, i] = djm[:, ix] * X[:, ix2] + djm[:, ix2] * X[:, ix]
#                     i += 1
#             return K
#         else:
#             bs_error_abort(f"covars cannot be {covars}!")
#     else:
#         bs_error_abort("X must be a vector or matrix")
#


def f_infty_BLP(Y: np.ndarray, Sigma: np.ndarray):
    pass
