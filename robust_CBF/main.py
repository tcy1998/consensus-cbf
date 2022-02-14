import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from Unicycle_dynamic import Unicycle_dynamic, Unicycle_Environment
from Cruise_dynamic import Cruise_Environment, Cruise_Dynamics
from Uncicyle_CBF import naive_CBF

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MPPI_control:

    def __init__(self):
        # self.mdl = Cruise_Dynamics()
        self.mdl = Unicycle_dynamic()
        self.d = self.mdl.d
        self.m = self.mdl.m
        self.K = self.mdl.K
        self.T = 50
        
        # 0. Initial control
        self.u_ts = torch.Tensor(np.zeros((self.T, self.m)))

        # 1.Initial State 
        self.x = torch.Tensor(np.zeros(self.d))

        # 2. Hyper parameter
        self.mu = torch.Tensor(np.zeros(self.m))
        self.sigma = torch.Tensor(np.eye(self.m))*0.1

        self.Lambda = 1
        self.control_limit = 100

    def control(self,x):
        print(x.size())
        self.x = x.view(self.d)

        # Step 0. Initilize the signals: (1) random explroation noise; (2) sum of cost, (3) state values
        x_init_dK = torch.transpose(self.x.repeat(self.K, 1), 0, 1) # Size (d, K)
        eps_TmK = torch.randn(self.T, self.m, self.K) # Size (T, m, K)
        S_K = torch.Tensor(np.zeros(self.K)) # Size (K)
        u_Tm = self.u_ts # Size (T, m)
        x_K = x_init_dK  # Size (d, K)

        for t in range(self.T):
            u_t = u_Tm[t]
            u_K = torch.transpose(u_t.repeat(self.K, 1), 0, 1)
            eps_mK = eps_TmK[t]
            # u_K = torch.clamp(u_K + eps_mK, -self.control_limit, self.control_limit)
            u_K = u_K + eps_mK
            
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
        # X = []
        # x = self.x.view(self.d, 1)
        # for t in range(self.T):
        #     X.append(x.view(-1))
        #     u = U_T[t].view(self.m, 1)
        #     # print(x.size(), u.repeat(1,self.K).size())
        #     x_next = self.mdl.dynamic(x, u.repeat(1,self.K))
        #     x = x_next
        # X = torch.stack(X)

        # Step 5. Update the controls
        self.u_ts[:-1] = U_T[1:]
        self.u_ts[-1] = torch.Tensor(np.zeros((self.m)))

        # Step 6. return numpy array
        U_np = self.u_ts.data.numpy()
        # X_np = X.data.numpy()
        X_np = []

        return U_np, X_np



if __name__ == "__main__":

    mppi = MPPI_control()
    # plant = Cruise_Environment()
    plant = Unicycle_Environment()
    safe_control = naive_CBF()
    time_steps = 500

    obs = plant.reset()

    x_s, y_s, z_s = [], [], []

    Use_CBF = True

    for t in range(time_steps):
        # print(obs)
        [x, y, z] = obs
        dist =(plant.target_pos_x - x)**2 + (plant.target_pos_y - y)**2
        # U_np, X_np = mppi.control(obs)
        # u = torch.Tensor(U_np[0])

        U_np = [[-10.0 * dist.data.numpy(), -10 * z.data.numpy()], [0,0]]
        u = torch.Tensor(U_np[0])

        print(u,obs,U_np[0])
        if Use_CBF == True:
            u += safe_control.CBF(obs.data.numpy(), U_np[0])
        obs = plant.step(u)
        
        if dist < 0.09:
            print(dist, x, y, t)
            break
        x_s.append(x)
        y_s.append(y)
        z_s.append(z)

    # t = np.linspace(0,time_steps*0.02,num=time_steps)
    # plt.plot(t, x_s)
    # plt.plot(t, y_s)
    # plt.plot(t, z_s)
    plt.plot(x_s, y_s)
    circle1 = plt.Circle((plant.obstacle_x, plant.obstacle_y), plant.r, color='r')
    plt.gca().add_patch(circle1)
    plt.show()
