import numpy as np
import torch
import math

class Unicycle_dynamic:
    def __init__(self):
        self.dt = 0.02      # The frequency set to 50hz
        self.d = 3          # The imension of state
        self.m = 2          # The dimension of control input
        self.K = 10000        # The number of sample
        self.T = 20         # The length of timestep

        self.mu = 0.0     # The mean of the noise 
        self.sigma = 1.0  # The sigma function of the Brownian Motion
        self.obstacle_type = 'sin'
        self.use_robust = False

        if self.obstacle_type == 'circle':
            self.target_pos_x, self.target_pos_y = 4.0, 4.0
            self.obstacle_x, self.obstacle_y = 2.2, 2.0
            self.r = 0.5
            self.control_limit = 20

        if self.obstacle_type == 'sin':
            self.target_pos_x, self.target_pos_y = 4.0, 0.5
            self.control_limit = 20
            self.width = 1

    def dynamic(self, X, U):
        '''
        State transition model
        input:
        x is tensor with size (d, K) where K is the number of sample trajectories.
        u is tensor with size (m, K)
        return:
        x_next is tensor with size(d, K) 
        '''
        dW = torch.Tensor(np.random.normal(self.mu, self.dt,\
            size=(self.d, 1))) # The noise of the dynamic size is (d,K)
        x_d = torch.cos(X[2]) * U[0]
        y_d = torch.sin(X[2]) * U[0]    #element wise product
        w_d = U[1]
        # print(x_d.size(), U[0].size())
        X_d = torch.vstack((x_d, y_d, w_d))
        X_new = X_d * self.dt #+ dW * self.sigma

        return X + X_new

    def cost_f(self, X, U):
        '''Running cost
        input:
        x is tensor with size (d, K) where K is the number of sample trajectories.
        u is tensor with size (m, K)
        return:
        cost is tensor with size K 
        '''
        if self.obstacle_type == 'circle':
            C1 = (self.target_pos_x - X[0])**2 + (self.target_pos_y - X[1])**2
            C2 = (self.obstacle_x - X[0])**2 + (self.obstacle_y - X[1])**2
            Ind2 = torch.where(C2<self.r**2, torch.ones(C1.size()), torch.zeros(C1.size()))

            return 1 * C1 #+ 1000 * Ind2
        
        if self.obstacle_type == 'sin':
            C1 = (self.target_pos_x - X[0])**2 + (self.target_pos_y - X[1])**2
            Ind1 = torch.where(X[1]<torch.sin(0.5*np.pi*X[0]),  torch.ones(C1.size()), torch.zeros(C1.size()))
            Ind2 = torch.where(X[1]>torch.sin(0.5*np.pi*X[0])+self.width,  torch.ones(C1.size()), torch.zeros(C1.size()))

            return 1 * C1 + 10000 * (Ind1 + Ind2)

    def terminal_f(self, x, u):
        C = (self.target_pos_x - x[0])**2 + (self.target_pos_y - x[1])**2
        return 100*C

class Unicycle_Environment(Unicycle_dynamic):

    def __init__(self):
        Unicycle_dynamic.__init__(self)
        self.x = torch.Tensor(np.zeros((self.d, 1)))
        

    def reset(self):
        self.x = torch.Tensor(np.array([[0.0],[0.5],[0]]))
        return self.x

    def step(self, u):
        # u = torch.clamp(u, -self.control_limit, self.control_limit)
        x_next = self.dynamic(self.x, u)
        self.x = x_next
        return x_next
