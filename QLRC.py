""" defines a class for QLRC models """


class QLRCModel:
    def __init__(self, f0, f1, f2, f_infty=None):
        self.f0, self.f1, self.f2 = f0, f1, f2
        if f_infty is not None:
            self.f_infty = f_infty

    def take_data(self, Y):
        self.Y = Y
        self.K = self.create_regressors(Y)

    def create_regressors(self, Y):
        return f2(Y)

    def estimate(self):
        pass

    def simulate(self):
        if self.f_infty is not None:
            pass