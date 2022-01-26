import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import cvxpy as cp

class cruise_robust_CBF:

    def __init__(self):
        self.d = 3  #state dimension
        self.m = 1  #control input dimension
        self.alpha = 1000   #hyperparameter for cbf
        self.M = 1650
    
    def cbf_h(self, x):     #cbf
        return x[2] - 1.8 * x[0]
    def partial_h (self, x):        #partial derivative of cbf
        return np.matrix([-1.8], [0], [1])

    def g_x(self, x):
        return np.matrix([1/self.M], [0.0], [0.0])

    def QP(self, x, u):     #using cvxpy to solve qp
        h = self.cbf_h(x) + self.partial_h(x) * u
        P = np.identity(self.m)
        q = np.zeros(self.m)
        g = - self.partial_h(x) * self.g_x(x)


class Cruise_Environment:

    def __init__(self):
        self.f_0 = 0.1
        self.f_1 = 5
        self.f_2 = 0.25
        self.M = 1650
        self.dt = 0.02

        self.mu = 0
        self.sigma = 0.1

    def F_r(self, x):
        return self.f_0 + self.f_1 * x[0] + self.f_2 * x[0] ** 2
        

    def dynamic(self, x, u):
        A = np.matrix([-self.F_r(x)/self.M], [0.0], [x[1]-x[0]])
        B = np.matrix([1/self.M], [0.0], [0.0])
        dW = np.random.normal(self.mu, self.sigma, size=(3,1))

        dx = A + np.matmul(B, u) + dW
        return x + dx * self.dt
