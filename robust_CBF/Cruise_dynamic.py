import numpy as np
import torch

class Cruise_Dynamics:

    def __init__(self):
        self.f_0 = 0.1
        self.f_1 = 5
        self.f_2 = 0.25
        self.M = 1650
        self.dt = 0.02

        self.mu = 0
        self.sigma = 1

        self.d = 3
        self.m = 1

        self.K = 500

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
        speed_tgt = 22
        C1 = (x[0]- speed_tgt)**2
        # C2 = (4.0-x[0])**2 + (4.0-x[1])**2
        C2 = u**2


        # 1. Possition Contraint with Indicator Functions
        distance_from_center = (x[0]**2 + x[1]**2)**0.5
        Ind0 = torch.where(distance_from_center > 2.125, torch.ones(C1.size()), torch.zeros(C1.size()))
        Ind1 = torch.where(distance_from_center < 1.875, torch.ones(C1.size()), torch.zeros(C1.size()))
        return C1 *100


    def terminal_f(self, x, u):
        return 0

class Cruise_Environment(Cruise_Dynamics):

    def __init__(self):
        Cruise_Dynamics.__init__(self)
        self.x = torch.Tensor(np.zeros((self.d, 1)))

    def reset(self):
        # self.x = torch.Tensor(np.zeros((self.d, 1)))
        self.x = torch.Tensor(np.array([[18], [10], [150]]))
        return self.x
    
    def step(self, u):
        # u = torch.clamp(u, -100, 100)
        x_next = self.dynamic(self.x, u.view(self.m, 1))
        self.x = x_next
        return x_next