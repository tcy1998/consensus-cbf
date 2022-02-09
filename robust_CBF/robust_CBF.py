import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import cvxpy as cp
import torch
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


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

        self.K = 2500

        self.v_desire = 22

    def F_r(self, x):       #Used in dynamics
        F_xn = []
        for x_n in torch.transpose(x, 0, 1).split(1):
            fx_n = self.f_0 + self.f_1 * x_n[0][0] + self.f_2 * x_n[0][0] ** 2
            F_xn.append(-fx_n.item()/self.M)
        return F_xn
        
    def G_r(self, x):
        G_xn = []
        for x_n in torch.transpose(x, 0, 1).split(1):
            g_xn = x_n[0][1] - x_n[0][2]
            G_xn.append(g_xn.item())
        return G_xn

    def dynamic(self, x, u):        #dynamic updates
        # print(x.size()[1])
        A = torch.Tensor([self.F_r(x), [0.0]* x.size()[1], self.G_r(x)])
        B = torch.Tensor(np.array([[1/self.M], [0.0], [0.0]]))
        dW = torch.Tensor(np.random.normal(self.mu, self.sigma, size=(3,1)))

        dx = A + torch.mm(B, u) + dW
        return x + self.dt * dx
    
    def cost_f(self, x, u):
        '''
        Running cost
        input:
        x is tensor with size (d, K) where K is the number of sample trajectories.
        u is tensor with size (m, K)
        return:
        cost is tensor with size K 
        '''
        # 0. Speed Error Cost
        speed_tgt = 1.0 
        C1 = (x[2]- speed_tgt)**2
        # C2 = (4.0-x[0])**2 + (4.0-x[1])**2
        C2 = 0


        # 1. Possition Contraint with Indicator Functions
        distance_from_center = (x[0]**2 + x[1]**2)**0.5
        Ind0 = torch.where(distance_from_center > 2.125, torch.ones(C1.size()), torch.zeros(C1.size()))
        Ind1 = torch.where(distance_from_center < 1.875, torch.ones(C1.size()), torch.zeros(C1.size()))
        return C1 + C2 + 1000*(Ind0+Ind1)


    def terminal_f(self, x, u):
        return 0

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


class MPPI_control:

    def __init__(self):
        self.mdl = Cruise_Environment()
        self.d = self.mdl.d
        self.m = self.mdl.m
        self.K = self.mdl.K
        self.T = 20
        
        # 0. Initial control
        self.u_ts = torch.Tensor(np.zeros((self.T, self.m)))

        # 1.Initial State 
        self.x = torch.Tensor(np.zeros(self.d))

        # 2. Hyper parameter
        self.mu = torch.Tensor(np.zeros(self.m))
        self.sigma = torch.Tensor(np.eye(self.m))

        self.Lambda = 1.0
        self.control_limit = 10

    def control(self,x):

        self.x = x.view(self.d)

        # Step 0. Initilize the signals: (1) random explroation noise; (2) sum of cost, (3) state values
        x_init_dK = torch.transpose(self.x.repeat(self.K, 1), 0, 1) # Size (d, K)
        eps_TmK = torch.randn(self.T, self.m, self.K) # Size (T, m, K)
        # eps_TmK = torch.normal(mean=self.mu, std=self.sigma, size=(self.T, self.m, self.K))
        S_K = torch.Tensor(np.zeros(self.K)) # Size (K)
        u_Tm = self.u_ts # Size (T, m)
        x_K = x_init_dK  # Size (d, K)

        for t in range(self.T):
            u_t = u_Tm[t]
            u_K = torch.transpose(u_t.repeat(self.K, 1), 0, 1)
            eps_mK = eps_TmK[t]
            u_K = torch.clamp(u_K + eps_mK, -self.control_limit, self.control_limit)
            
            x_K_next = self.mdl.dynamic(x_K, u_K)

            C_K = self.mdl.cost_f(x_K, u_K)
            S_K = S_K + C_K + self.Lambda*torch.mm(torch.mm(u_t.view(1, self.m), self.sigma),eps_mK)

            x_K = x_K_next
        
        S_K += self.mdl.terminal_f(x_K, u_K)

        # Step 2. Calculate weights
        rho = torch.min(S_K)
        omega_tilde = torch.exp(-(S_K - rho)/self.Lambda)
        eta = torch.sum(omega_tilde)

        # Step 3. Calculate control input.
        U_T = self.u_ts
        print(np.shape(eps_TmK),np.shape(omega_tilde))
        U_T = U_T + torch.sum(eps_TmK*omega_tilde, 2)/eta

        # Step 4. Forward Calculation
        X = []
        x = self.x.view(self.d, 1)
        for t in range(self.T):
            X.append(x.view(-1))
            u = U_T[t].view(self.m, 1)
            # print(x.size())
            x_next = self.mdl.dynamic(x, u)
            x = x_next
        X = torch.stack(X)

        # Step 5. Update the controls
        self.u_ts[:-1] = U_T[1:]
        self.u_ts[-1] = torch.Tensor(np.zeros((self.m)))

        # Step 6. return numpy array
        U_np = self.u_ts.data.numpy()
        X_np = X.data.numpy()

        return U_np, X_np

class Environment(Cruise_Environment):

    def __init__(self):
        Cruise_Environment.__init__(self)
        self.x = torch.Tensor(np.zeros((self.d, 1)))

    def reset(self):
        self.x = torch.Tensor(np.zeros((self.d, 1)))
        return self.x
    
    def step(self, u):
        u = torch.clamp(u, -10, 10)
        x_next = self.dynamic(self.x, u.view(self.m, 1))
        self.x = x_next
        return x_next

if __name__ == "__main__":

    mppi = MPPI_control()
    plant = Environment()

    obs = plant.reset()

    x_s, y_s, z_s = [], [], []

    for t in range(500):
        U_np, X_np = mppi.control(obs) 
        u = torch.Tensor(U_np[0])
        obs = plant.step(u)

        [x, y, z] = obs

        x_s.append(x)
        y_s.append(y)
        z_s.append(z)

    t = np.linspace(0,500*0.02,num=500)
    plt.plot(x_s, t, 'b+')
    plt.plot(y_s, t)
    plt.plot(z_s, t)
    plt.show()
