import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import cvxpy as cp

class Cruise_Environment:

    def __init__(self):
        self.f_0 = 0.1
        self.f_1 = 5
        self.f_2 = 0.25
        self.M = 1650
        self.dt = 0.02

        self.mu = 0
        self.sigma = 0.1

        self.d = 3
        self.m = 1

        self.v_desire = 22

    def F_r(self, x):       #Used in dynamics
        return self.f_0 + self.f_1 * x[0] + self.f_2 * x[0] ** 2
        

    def dynamic(self, x, u):        #dynamic updates
        A = np.matrix([-self.F_r(x)/self.M], [0.0], [x[1]-x[0]])
        B = np.matrix([1/self.M], [0.0], [0.0])
        dW = np.random.normal(self.mu, self.sigma, size=(3,1))

        dx = A + np.matmul(B, u) + dW
        return x + dx * self.dt
    
    def cost_f(self, x, u):
        return (1/2)* u ** 2

class cruise_robust_CBF:

    def __init__(self):
        self.mdl = Cruise_Environment
        self.d = self.mdl.d  #state dimension
        self.m = self.mdl.m  #control input dimension
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

        u_opt = cp.Variable(self.m)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x,P)+q.T @ u_opt)
                        [g @ x <= h])
        prob.solve()

        return u_opt.value


class MPPI:

    def __init__(self):
        self.mdl = Cruise_Environment
        self.d = self.mdl.d
        self.m = self.mdl.m
        self.K = 2500
        self.T = 200

        self.u_t_init = np.zeros((self.T, self.m))
        self.x_t_init = np.zeros(self.d)

        self.mu = np.zeros(self.m)
        self.sigma = np.eye(self.m)

        self.Lambda = 1.0
        self.control_limit = 10

    def control(self,x):
        u_Tm = self.u_t_init
        random_noise = np.random.multivariate_normal(self.mu, self.sigma, (self.K, self.T))
        x_init = np.transpose(self.repeat(self.K, 1))
        x_t = x_init

        for t in range(self.T):
            u_t = u_Tm[t]
            u_k = np.transpose(u_t.repeat(self.K, 1))
            random_n = random_noise[:,t]
            u_k = np.clip(u_k + random_n, -self.control_limit, self.control_limit)
            
            x_t_next = self.mdl.dynamic(x_t, u_k)

            C_K = self.mdl.cost_f(x_t_next, u_k)
            S_K = S_K + C_K + self.Lambda * np.matmul(np.matmul(u_t,))





