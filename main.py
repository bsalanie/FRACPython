""" estimates via 2SLS and corrected 2SLS """
import numpy as np

from QLRC import QLRCModel

from BLP_basic import f0_BLP, f1_BLP, A_star_BLP, A2_BLP, A33_BLP, f_infty_BLP, simulated_mean_shares_

if __name__ == "__main__":

    nmarkets = 50
    nproducts = 3
    nx = 2
    ndraws = 10000

    X = np.random.normal(size=(nmarkets, nproducts, nx))
    beta_bar = np.array([-2.0, 3.0])
    xi = np.random.normal(size=(nmarkets, nproducts))
    sigmas = np.array([0.2, 0.4])
    eps = np.random.normal(size=(nmarkets, nx, ndraws))
    n_Y = nproducts + nx
    Y = np.zeros((nmarkets, nproducts, n_Y))
    for t in range(nmarkets):
        X_t, eps_t, xi_t = X[t, :, :], eps[t, :, :], xi[t, :]
        Xsig_eps = np.outer(X_t[:, 0], sigmas[0] * eps_t[0, :])
        for ix in range(1, nx):
            Xsig_eps += (np.outer(X_t[:, ix], sigmas[ix] * eps_t[ix, :]))
        utils = Xsig_eps + (X_t @ beta_bar + xi_t).reshape((-1, 1))
        shares = simulated_mean_shares_(utils)
        Y[t, :, :nproducts] = np.tile(shares, (nproducts, 1))
        Y[t, :, nproducts:] = X_t

    n_betas = n_Sigma = nx
    Z = X

    model = QLRCModel(Y, A_star_BLP, f1_BLP, n_betas, n_Sigma, Z, f_0=f0_BLP,
                      args=[nproducts, A2_BLP, A33_BLP])
    model.fit()
    # model.predict(f_infty_BLP)
    # model.fit_corrected()

    model.print()
