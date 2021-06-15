""" estimates via 2SLS and corrected 2SLS """

from QLRC import QLRCModel

def f0():
    pass

def f1():
    pass

def f2():
    pass

def f_infty():
    pass



if __name__ == "__main__":
    model = QLRCModel(f0, f1, f2)
    model.estimate()
    model.simulate(f_infty)
    model.corrected_estimate()


