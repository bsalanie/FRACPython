""" estimates via 2SLS and corrected 2SLS """
import numpy as np
from QLRC import QLRCModel

from BLP_basic import f0_BLP, f1_BLP, K_BLP_diag, f_infty_BLP, d_BLP, t_BLP



if __name__ == "__main__":
    model = QLRCModel(f0_BLP, f1_BLP, d=d_BLP, t=t_BLP, K=K_BLP_diag)
    model.estimate()
    model.simulate(f_infty)
    model.corrected_estimate()


