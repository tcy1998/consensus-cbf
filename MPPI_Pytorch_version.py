import numpy as np 
import torch
import matplotlib.pyplot as plt



class TransitionNCostMdl:

    def __init__(self):
        '''
        Initilize a 2D double integrator model for state transition and cost function 
        '''
        self.A = torch.Tensor(np.array([[1, 0, 0.01, 0],[0, 1, 0, 0.01],[0, 0, 1, 0],[0, 0, 0, 1]]))
        self.B = torch.Tensor(np.array([[0, 0],[0, 0],[0.01, 0],[0, 0.01]]))
        self.d = 4 # state dim.
        self.m = 2 # control dim.


    def F(self, x, u):
        '''
        State transition model
        input:
        x is tensor with size (d, K) where K is the number of sample trajectories.
        u is tensor with size (m, K)
        return:
        x_next is tensor with size(d, K) 
        '''
        x_next = torch.mm(self.A, x) + torch.mm(self.B, u)
        return x_next


    def C(self, x, u):
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
        C1 = ((x[2]**2 + x[3]**2)**0.5 - speed_tgt)**2
        C2 = (1.0-x[0])**2 + (1.0-x[1])**2


        # 1. Possition Contraint with Indicator Functions
        distance_from_center = (x[0]**2 + x[1]**2)**0.5
        Ind0 = torch.where(distance_from_center > 2.125, torch.ones(C1.size()), torch.zeros(C1.size()))
        Ind1 = torch.where(distance_from_center < 1.875, torch.ones(C1.size()), torch.zeros(C1.size()))
        return C1 + C2 + 1000*(Ind0+Ind1)


    def Phi(self, x, u):
        '''
        Terminal cost
        '''
        return 0





class ModelPredictivePathIntegral:
    '''
    Implement the MPPI control.
    '''

    def __init__(self):
        self.mdl = TransitionNCostMdl()
        self.d = self.mdl.d # size of state dimension
        self.m = self.mdl.m # size of control dimension
        self.K = 2500 # Number of samples
        self.T = 200  # Length of horizon

        # 0. Control signal to be updated.
        self.u_ts = torch.Tensor(np.zeros((self.T, self.m)))

        # 1. Current state
        self.x = torch.Tensor(np.zeros(self.d))

        # 2. Random noise and softmax function temp. parameter 
        self.Sigma = torch.Tensor(np.eye(self.m))
        self.Lambda = 1.0  



    def control(self, x):
        '''
        Determine the control using the random samples.
        '''

        self.x = x.view(self.d)

        # Step 0. Initilize the signals: (1) random explroation noise; (2) sum of cost, (3) state values
        x_init_dK = torch.transpose(self.x.repeat(self.K, 1), 0, 1) # Size (d, K)
        eps_TmK = torch.randn(self.T, self.m, self.K) # Size (T, m, K)
        S_K = torch.Tensor(np.zeros(self.K)) # Size (K)
        u_Tm = self.u_ts # Size (T, m)
        x_K = x_init_dK  # Size (d, K)

        # Step 1. Roll multiple trajectories. 
        for t in range(self.T):
            # 1-1. Sample K random disturbance.
            u_t = u_Tm[t]
            u_K = torch.transpose(u_t.repeat(self.K, 1), 0, 1)
            eps_mK = eps_TmK[t]
            u_K = torch.clamp(u_K + eps_mK, -10, 10)

            # 1-2. Get transitions.
            x_K_next = self.mdl.F(x_K, u_K)

            # 1-3. Get running cost
            C_K = self.mdl.C(x_K, u_K)
            S_K = S_K + C_K + self.Lambda*torch.mm(torch.mm(u_t.view(1, self.m), self.Sigma),eps_mK)

            # 1-4. Update the state
            x_K = x_K_next

        S_K = S_K + self.mdl.Phi(x_K, u_K)

        # Step 2. Determine Weights using the exponential distirbution form.
        rho = torch.min(S_K)
        omega_tilde = torch.exp(-(S_K - rho)/self.Lambda)
        eta = torch.sum(omega_tilde)
        

        # Step 3. Calculate control input.
        U_T = self.u_ts
        print(np.shape(eps_TmK),np.shape(omega_tilde))
        U_T = U_T + torch.sum(eps_TmK*omega_tilde, 2)/eta

        # Step 4. Simulate with U_T
        X = []
        x = self.x.view(self.d, 1)
        for t in range(self.T):
            X.append(x.view(-1))
            u = U_T[t].view(self.m, 1)
            x_next = self.mdl.F(x, u)
            x = x_next            
        X = torch.stack(X)
        
        # Step 5. Update the control
        self.u_ts[:-1] = U_T[1:]
        self.u_ts[-1] = torch.Tensor(np.zeros((self.m)))

        # Step 6. Return Numpy array 
        U_np = self.u_ts.data.numpy()
        X_np = X.data.numpy()

        return U_np, X_np
            

class Environment(TransitionNCostMdl):

    def __init__(self):
        TransitionNCostMdl.__init__(self)
        self.x = torch.Tensor(np.zeros((self.d, 1)))

    def reset(self):
        self.x = torch.Tensor(np.zeros((self.d, 1)))
        return self.x

    def step(self, u):
        u = torch.clamp(u, -10, 10)
        x_next = self.F(self.x, u.view(self.m, 1))
        self.x = x_next
        return x_next




if __name__ == "__main__":

    plant = Environment()
    mppi = ModelPredictivePathIntegral()

    obs = plant.reset()

    x_s = []
    y_s = []
    vx_s = []
    vy_s = []

    for t in range(500):
        #print(t)
        U_np, X_np = mppi.control(obs)
        u = torch.Tensor(U_np[0])
        #print(u)
        obs = plant.step(u)

        x = float(obs[0])
        y = float(obs[1])
        vx = float(obs[2])
        vy = float(obs[3])

        x_s.append(x)
        y_s.append(y)
        vx_s.append(vx)
        vy_s.append(vy)
        print(t, x, y, vx, vy)


    plt.plot(x_s, y_s, 'b+')
    plt.show()