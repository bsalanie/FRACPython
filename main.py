""" estimates via 2SLS and corrected 2SLS """
import numpy as np
import scipy.linalg as spla

from QLRC import QLRCModel, least_squares_proj

from BLP_basic import f0_BLP, f1_BLP, A_star_BLP, A2_BLP, A33_BLP, K_BLP,\
    f_infty_BLP, simulated_mean_shares_

from stats_utils import reg_nonpar_fit, flexible_reg

if __name__ == "__main__":

    n_markets = 100
    n_products = 10
    nx = 2
    n_draws = 10000

    X = np.random.normal(size=(n_markets, n_products, nx))
    # the first covariate is the constant
    X[:, :, 0] = 1.0
    beta_bar = np.array([0.0, 1.0])
    xi = np.random.normal(size=(n_markets, n_products))
    sigmas = np.array([0.5])
    # no random coefficient on the constant
    eps = np.random.normal(size=(n_markets, nx - 1, n_draws))
    n_Y = n_products + nx
    Y = np.zeros((n_markets, n_products, n_Y))
    K = np.zeros((n_markets, n_products, nx-1))
    for t in range(n_markets):
        X_t, eps_t, xi_t = X[t, :, :], eps[t, :, :], xi[t, :]
        Xsig_eps = np.zeros((n_products, n_draws))
        for ix in range(1, nx):
            Xsig_eps += (np.outer(X_t[:, ix], sigmas[ix-1] * eps_t[ix-1, :]))
        utils = Xsig_eps + (X_t @ beta_bar + xi_t).reshape((-1, 1))
        shares = simulated_mean_shares_(utils)
        # Y[t, i, :n_products] is the vector of market shares
        Y[t, :, :n_products] = np.tile(shares, (n_products, 1))
        # Y[t, i, :n_products] is the vector of market shares
        Y[t, :, n_products:] = X_t
        for ix in range(nx):
            xtix = X_t[:, ix]
            eSix = xtix @ shares
            K[t, :, ix-1] = xtix*(xtix/2.0-eSix)


    n_betas =  nx
    n_Sigma = nx - 1
    Z = X

    model = QLRCModel(Y, A_star_BLP, f1_BLP, n_betas, n_Sigma, Z, f_0=f0_BLP,
                      # K = K_BLP,
                      args=[n_products, A2_BLP, A33_BLP])
    model.fit()

    print(f"True betas: {beta_bar}")
    print(f"True Sigma: {sigmas**2}")
    # model.predict(f_infty_BLP)
    # model.fit_corrected()

    model.print()

    n_points = n_markets * n_products
    f_0 = np.zeros((n_markets, n_products))
    for t in range(n_markets):
        f_0[t, :] = f0_BLP(Y[t, :, :], [n_products])
    f0r = f_0.reshape(n_points)
    Xr = np.zeros((n_points, nx))
    Kr = np.zeros((n_points, nx-1))
    Kfit = np.zeros((n_points, nx-1))
    for ix in range(nx):
        Xr[:, ix] = X[:, :, ix].reshape((n_points))
    for ix in range(nx-1):
        Kr[:, ix] = K[:, :, ix].reshape((n_points))
    for ix in range(nx-1):
        # Kfit[:, ix]  = reg_nonpar_fit(Kr[:, ix], Xr)
        Kfit[:, ix]  = flexible_reg(Kr[:, ix], Xr,  mode='2')

    rhs = np.concatenate((Xr, Kfit), axis=1)
    coeffs, _, _, _ = spla.lstsq(rhs, f0r)
    print(coeffs)







