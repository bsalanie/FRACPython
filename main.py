""" estimates via 2SLS and corrected 2SLS """
import numpy as np

from QLRC import QLRCModel

from BLP_basic import f0_BLP, f1_BLP, K_BLP_diag, f_infty_BLP



if __name__ == "__main__":

    model = QLRCModel(Y, A_star_BLP, f0_BLP, f1_BLP,K=K_BLP_diag)
    model.fit()
    model.predict(f_infty_BLP)
    model.fit_corrected()

    model.print()


