import numpy as np
import torch
from Unicycle_dynamic import Unicycle_dynamic
import cvxpy as cp
# from cvxopt import solvers, matrix

class naive_CBF:
    def __init__(self):
        self.mdl = Unicycle_dynamic()
        self.alpha = 1.0
        self.use_robust = False
        self.obstacle_type = self.mdl.obstacle_type

    def g_x(self, X):   # 3x2 matrix
        return np.matrix([[np.cos(X[2]).item(), 0.0], \
                        [np.sin(X[2]).item(), 0.0], \
                        [0.0, 1.0]])
    
    def cbf_h(self,X):  # fixed value
        if self.obstacle_type == 'circle':
            value = (-self.mdl.obstacle_x + X[0])**2 +\
                    (-self.mdl.obstacle_y + X[1])**2 -\
                    self.mdl.r**2
            if self.use_robust == False:
                return self.alpha * value
            else:
                return self.alpha * value + 0.0001
        if self.obstacle_type == 'sin':
            h1 = X[1] - np.sin(2*np.pi*X[0])
            h2 = self.mdl.width - X[1] + np.sin(2*np.pi*X[0])
            # print(h1, h2)
            return np.matrix([[h1[0]],[h2[0]]])

    def partial_h(self, X): # 1x3 vector
        if self.obstacle_type == 'circle':
            x1 = -self.mdl.obstacle_x + X[0]
            x2 = -self.mdl.obstacle_y + X[1]
            return np.matrix([[2*x1, 2*x2, 0]])

        if self.obstacle_type == 'sin':
            x1 = - 2*np.pi*np.cos(2*np.pi*X[0])
            x2 = 2*np.pi*np.cos(2*np.pi*X[0])
            # print(x1, x2)
            return np.matrix([[x1[0], 1, 0], [x2[0], -1, 0]])

    def CBF(self,X,U):
        P = np.matrix([[1.0,0.0],[0.0,1.0]])
        q = np.matrix([[0.0], [0.0]])
        g = np.matmul(self.partial_h(X), self.g_x(X))   # N*2 matrix
        h = self.cbf_h(X) + np.matmul(g, U.reshape(2,1))
        # print( self.cbf_h(X), g@U)
        # print(np.shape(h))
        h = np.asarray(h)
        H = np.squeeze(h)
        # print(np.shape(g), np.shape(H), np.shape(h), np.shape(U))


        u_opt = cp.Variable(self.mdl.m)
        prob = cp.Problem(cp.Minimize(cp.quad_form(u_opt,P)),
                        [-g @ u_opt <= H])
        prob.solve()

        return u_opt.value

        

if __name__ == "__main__":
    b = naive_CBF()
    Initial_state = np.array([[1.5],[1.9],[np.pi/10]])
    u_init = np.array([[1],[1]])
    b.CBF(Initial_state, u_init)