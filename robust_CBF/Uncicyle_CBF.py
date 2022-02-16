import numpy as np
import torch
from Unicycle_dynamic import Unicycle_dynamic
import cvxpy as cp
from cvxopt import solvers, matrix

class naive_CBF:
    def __init__(self):
        self.mdl = Unicycle_dynamic()
        self.alpha = 10
    
    def cbf_h(self,X):  # fixed value
        value = (-self.mdl.obstacle_x + X[0])**2 +\
                (-self.mdl.obstacle_y + X[1])**2 -\
                self.mdl.r**2
        return self.alpha * value

    def g_x(self, X):   # 3x2 matrix
        return np.matrix([[np.cos(X[2]), 0.0], \
                          [np.sin(X[2]), 0.0], \
                          [0.0, 1.0]])

    def partial_h(self, X): # 1x3 vector
        x1 = -self.mdl.obstacle_x + X[0]
        x2 = -self.mdl.obstacle_y + X[1]
        return np.matrix([[2*x1, 2*x2, 0]])

    def CBF(self,X,U):
        P = np.matrix([[1.0,0.0],[0.0,1.0]])
        q = np.matrix([[0.0], [0.0]])
        g = np.matmul(self.partial_h(X), self.g_x(X))   #1*2 vector
        h = self.cbf_h(X) + np.matmul(g,U)  #value
        u_opt = cp.Variable(self.mdl.m)
        prob = cp.Problem(cp.Minimize(cp.quad_form(u_opt,P)),
                        [-g @ u_opt <= h])
        prob.solve()
        # print(u_opt.value,g,h[0])
        # print("State", X[2])
        return u_opt.value


if __name__ == "__main__":
    b = naive_CBF()
    Initial_state = np.array([[1.5],[1.9],[np.pi/10]])
    u_init = np.array([[1],[1]])
    # b.g_x(Initial_state)
    b.CBF(Initial_state, u_init)