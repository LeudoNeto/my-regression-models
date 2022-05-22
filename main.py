import numpy as np
from scipy.interpolate import lagrange

class InterpolationRegression:
    def fit(self,x,y):
        self.poly = lagrange(x,y)
        self.coef_ = self.poly.coef[1::-1]
        self.intercept_ = self.poly.coef[0]

    def predict(self,x):
        return np.polyval(self.poly,x)