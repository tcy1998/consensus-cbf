import numpy as np
import torch
import cvxpy as cp
import time

from Unicycle_dynamic import Unicycle_dynamic

class SDP_CBF:
    def __init__(self):
        self.mdl = Unicycle_dynamic()
        self.d = self.mdl.d
        self.m = self.mdl.m
        self.mu = self.mdl.mu
        self.var = (self.mdl.sigma**2) * self.mdl.dt
        self.K = self.mdl.K

        self.use_robust = self.mdl.use_robust
        self.obstacle_type = self.mdl.obstacle_type
        self.alpha = 1.0

    def g_x(self, X):   # 3x2 matrix
        return np.matrix([[np.cos(X[2]).item(), 0.0], \
                        [np.sin(X[2]).item(), 0.0], \
                        [0.0, 1.0]])

    def partial_h(self, X): # 1x3 vector
        if self.obstacle_type == 'circle':
            x1 = -self.mdl.obstacle_x + X[0]
            x2 = -self.mdl.obstacle_y + X[1]
            return np.matrix([[2*x1, 2*x2, 0]])

        if self.obstacle_type == 'sin':
            x1 = - 0.5*np.pi*np.cos(0.5*np.pi*X[0])
            x2 = 0.5*np.pi*np.cos(0.5*np.pi*X[0])
            par1 = np.matrix([[x1, 1, 0]])
            par2 = np.matrix([[x2, -1, 0]])
            return par1, par2

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
            h1 = X[1] - np.sin(0.5*np.pi*X[0])
            h2 = self.mdl.width - X[1] + np.sin(0.5*np.pi*X[0])
            return h1, h2

    def SDP(self, X, U):
        U_delta = []        #return the Control input size (m,K)

        # start_time = time.time()
        if self.obstacle_type == 'circle':
            for i in range(self.K):
                state = X.T[i]      # 3x1 vector
                u = U.reshape(2,1)
                g_value = self.g_x(state)
                partialh_value = self.partial_h(state)
                a = np.matmul(partialh_value, g_value)
                b = self.cbf_h(state) + a @ u
                v = cp.Variable((self.m, self.m), PSD=True)
                m = cp.Variable((self.m, 1))
                constraints = [-1*a @ v @ a.T + a @ m >> -b, v >= 0]
                objective = cp.Minimize(cp.norm(v-np.identity(2),2)+cp.norm(m,2)) 
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.CVXOPT)
                mu = np.squeeze(m.value)
                var = v.value
                delta_u = np.random.multivariate_normal(mu.T, var)
                U_delta.append(delta_u)
                # print(delta_u, mu, var)
        if self.obstacle_type == 'sin':
            for i in range(self.K):
                state = X.T[i]      # 3x1 vector
                u = U.reshape(2,1)
                g_value = self.g_x(state)
                ph1, ph2 = self.partial_h(state)
                h1, h2 = self.cbf_h(state)
                a1 = ph1 @ g_value
                a2 = ph2 @ g_value
                b1 = h1 + a1 @ u
                b2 = h2 + a2 @ u
                v = cp.Variable((self.m, self.m), PSD=True)
                m = cp.Variable((self.m, 1))
                constraints = [-1*a1 @ v @ a1.T + a1 @ m >> -b1, v >= 0,
                               -1*a2 @ v @ a2.T + a2 @ m >> -b2 ]
                objective = cp.Minimize(cp.norm(v-np.identity(2),2)+cp.norm(m,2)) 
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.CVXOPT)
                mu = np.squeeze(m.value)
                var = v.value
                delta_u = np.random.multivariate_normal(mu.T, var)
                U_delta.append(delta_u)


        # print("--- %s seconds ---" % (time.time() - start_time))
        return U_delta