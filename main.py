import numpy as np
from scipy.interpolate import lagrange
from scipy.optimize import minimize

class InterpolationRegression:
    def fit(self,x,y):
        self.poly = lagrange(x,y)
        self.coef_ = self.poly.coef[1::-1]
        self.intercept_ = self.poly.coef[0]

    def predict(self,x):
        return np.polyval(self.poly,x)

class MyRegressionModel:
    def fit(self,x,y):
        x, y = zip(*sorted(zip(x,y))) #ordering x and y

        for pos,absciss in enumerate(x):
            if pos == 0:
                self.points = np.array([[absciss,y[0]]])
            elif pos == len(x)-1:
                self.points = np.vstack([self.points,np.array([[absciss,y[len(x)-1]]])])
            else:
                #self.points = np.vstack([self.points,np.array([[absciss,(y[pos-1]+y[pos]+y[pos+1])/3]])]) do the same, but the code below matches better with the model purpose
                def f(var):
                    y0 = var[0]
                    func = (y0 - y[pos-1])**2 + (y0 - y[pos])**2 + (y0 - y[pos+1])**2
                    return func
            
                self.points = np.vstack([self.points,np.array([[absciss,float(minimize(f,0).x)]])])

        self.poly = lagrange([point[0] for point in self.points],[point[1] for point in self.points])
        self.coef_ = self.poly.coef[1::-1]
        self.intercept_ = self.poly.coef[0]

    def predict(self,x):
        return np.polyval(self.poly,x)