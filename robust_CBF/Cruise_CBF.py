from Cruise_dynamic import Cruise_Environment, Cruise_Dynamics
import numpy as np
import cvxpy as cp


class naive_CBF:

    def __init__(self):
        self.mdl = Cruise_Dynamics
        self.d = self.mdl.d  #state dimension
        self.m = self.mdl.m  #control input dimension
        self.alpha = 1000   #hyperparameter for cbf
        self.M = 1650
    
    def cbf_h(self, x):     #cbf
        return x[2] - 1.8 * x[0]
    def partial_h (self, x):        #partial derivative of cbf
        return np.matrix([[-1.8], [0], [1]])

    def g_x(self, x):
        return np.matrix([[1/self.M], [0.0], [0.0]])

    def QP(self, x, u):     #using cvxpy to solve qp
        h = self.cbf_h(x) + self.partial_h(x) @ u
        P = np.identity(self.m)
        q = np.zeros(self.m)
        g = self.partial_h(x) * self.g_x(x)

        u_opt = cp.Variable(self.m)
        prob = cp.Problem(cp.Minimize(cp.quad_form(u_opt,P)+q.T @ u_opt),
                        [-g @ u_opt <= h])
        prob.solve()

        return u_opt.value