from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from Unicycle_dynamic import Unicycle_dynamic, Unicycle_Environment
from Cruise_dynamic import Cruise_Environment, Cruise_Dynamics
from Uncicyle_CBF import naive_CBF
from SDP_CBF import SDP_CBF
from numpy import savetxt
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MPPI_control:

    def __init__(self):
        self.mdl = Unicycle_dynamic()
        self.d = self.mdl.d
        self.m = self.mdl.m
        self.K = self.mdl.K
        self.T = self.mdl.T
        
        # 0. Initial control
        self.u_ts = torch.Tensor(np.zeros((self.T, self.m)))

        # 1.Initial State 
        self.x = torch.Tensor(np.zeros(self.d))

        # 2. Hyper parameter
        self.mu = torch.Tensor(np.zeros(self.m))
        self.sigma = torch.Tensor(np.eye(self.m))

        self.Lambda = 1
        self.control_limit = self.mdl.control_limit
        
        self.SDP_CBF = SDP_CBF()

    def mppi_control(self,x):
        self.x = x.view(self.d)

        # Step 0. Initilize the signals: (1) random explroation noise; (2) sum of cost, (3) state values
        x_init_dK = torch.transpose(self.x.repeat(self.K, 1), 0, 1) # Size (d, K)
        eps_TmK = torch.randn(self.T, self.m, self.K) # Size (T, m, K)
        S_K = torch.Tensor(np.zeros(self.K))            # Size (K)
        u_Tm = self.u_ts                                # Size (T, m)
        x_K = x_init_dK                                 # Size (d, K)
        state_sample = torch.Tensor(np.zeros((self.T, self.d, self.K)))

        for t in range(self.T):
            u_t = u_Tm[t]
            u_K = torch.transpose(u_t.repeat(self.K, 1), 0, 1)
            eps_mK = eps_TmK[t]
            
            u_K = torch.clamp(u_K + eps_mK, -self.control_limit, self.control_limit)
            u_K = u_K + eps_mK
            
            x_K_next = self.mdl.dynamic(x_K, u_K)
            # print(x_K_next.size())
            state_sample[t] = x_K_next

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
        # print(np.shape(eps_TmK),np.shape(omega_tilde))
        U_T = U_T + torch.sum(eps_TmK*omega_tilde, 2)/eta

        # Step 5. Update the controls
        self.u_ts[:-1] = U_T[1:]
        self.u_ts[-1] = torch.Tensor(np.zeros((self.m)))

        # Step 6. return numpy array
        U_np = self.u_ts.data.numpy()
        # X_np = X.data.numpy()
        X_np = state_sample.data.numpy()


        return U_np, X_np, S_K

    def cbf_control(self,x):
        self.x = x.view(self.d)

        # Step 0. Initilize the signals: (1) random explroation noise; (2) sum of cost, (3) state values
        x_init_dK = torch.transpose(self.x.repeat(self.K, 1), 0, 1) # Size (d, K)
        eps_TmK = torch.zeros(self.T, self.m, self.K)
        S_K = torch.Tensor(np.zeros(self.K))            # Size (K)
        u_Tm = self.u_ts                                # Size (T, m)
        x_K = x_init_dK                                 # Size (d, K)
        state_sample = torch.Tensor(np.zeros((self.T, self.d, self.K)))
        # Var, Cost = [], []

        for t in range(self.T):
            u_t = u_Tm[t]
            u_K = torch.transpose(u_t.repeat(self.K, 1), 0, 1)
            eps_mK  = self.SDP_CBF.SDP(x_K.data.numpy(), u_t.data.numpy())
            eps_mK = torch.tensor(np.transpose(eps_mK)).float()
            eps_TmK[t] = eps_mK
            
            u_K = torch.clamp(u_K + eps_mK, -self.control_limit, self.control_limit)
            u_K = u_K + eps_mK
            
            x_K_next = self.mdl.dynamic(x_K, u_K)
            state_sample[t] = x_K_next

            C_K = self.mdl.cost_f(x_K, u_K)
            S_K = S_K + C_K + self.Lambda*torch.mm(torch.mm(u_t.view(1, self.m), self.sigma),eps_mK)

            x_K = x_K_next
            # Var.append(var)
        
        S_K += self.mdl.terminal_f(x_K, u_K)
        

        # Step 2. Calculate weights
        rho = torch.min(S_K)
        omega_tilde = torch.exp(-(S_K - rho)/self.Lambda)
        eta = torch.sum(omega_tilde)

        # Step 3. Calculate control input.
        U_T = self.u_ts
        # print(np.shape(eps_TmK),np.shape(omega_tilde))
        U_T = U_T + torch.sum(eps_TmK*omega_tilde, 2)/eta

        # Step 5. Update the controls
        self.u_ts[:-1] = U_T[1:]
        self.u_ts[-1] = torch.Tensor(np.zeros((self.m)))

        # Step 6. return numpy array
        U_np = self.u_ts.data.numpy()
        # X_np = X.data.numpy()
        X_np = state_sample.data.numpy()

        return U_np, X_np, S_K

    def plot(self, state_x, state_y):
        if self.mdl.obstacle_type == 'circle':
            plt.plot(state_x, state_y)
            circle1 = plt.Circle((self.mdl.obstacle_x, self.mdl.obstacle_y),\
                self.mdl.r, color='r')
            plt.gca().add_patch(circle1)
            plt.draw()
            plt.pause(1)
            input("<Hit Enter>")
            plt.close()
        if self.mdl.obstacle_type == 'sin':
            plt.plot(state_x, state_y)
            x = np.arange(0,4,0.01)
            y = np.sin(0.5 * np.pi * x)
            plt.plot(x, y)
            plt.plot(x, y+self.mdl.width)
            plt.draw()
            plt.pause(1)
            input("<Hit Enter>")
            plt.close()


if __name__ == "__main__":

    mppi = MPPI_control()
    # plant = Cruise_Environment()
    plant = Unicycle_Environment()
    safe_control = naive_CBF()
    time_steps = 250

    obs = plant.reset()

    x_s, y_s, z_s = [], [], []
    v_s, w_s = [], []
    Sample = []
    S_cost = []

    Use_CBF = False
    Collect_sample = False


    for t in range(time_steps):
        if Use_CBF == True:        
            U_np, X_np, S_K = mppi.cbf_control(obs)
        else:
            U_np, X_np, S_K = mppi.mppi_control(obs)

        if Collect_sample == True:
            Sample.append(X_np)

        [x, y, z] = obs
        dist =(plant.target_pos_x - x)**2 + (plant.target_pos_y - y)**2
        if dist < 0.25**2: 
            print(dist, x, y, t)
            break

        u = torch.Tensor(U_np[0])
        obs = plant.step(u)         #Update dynamics
        
        print(t, x, y)
        
        #If the distance to the target is smaller than 0.3 stop

        x_s.append(x)
        y_s.append(y)
        z_s.append(z)
        v_s.append(u[0])
        w_s.append(u[1])
        S_cost.append(S_K)
        print(S_K)

    mppi.plot(x_s, y_s)
    
    # Get the current working directory
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # Print the type of the returned object
    print("os.getcwd() returns an object of type: {0}".format(type(cwd)))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(np.shape(Sample))
    print(np.shape(S_cost))

    # Collect the data
    # if Use_CBF == True:
    #     savetxt('robust_CBF/data_plot/A{}sample_{}steps_{}_CBF_{}.csv'.format(plant.K,plant.T,plant.obstacle_type,timestr), [x_s, y_s], delimiter=',')
    #     savetxt('robust_CBF/data_plot/C{}sample_{}steps_{}_CBF_{}.csv'.format(plant.K,plant.T,plant.obstacle_type,timestr), [z_s, v_s, w_s], delimiter=',')
    #     np.save('robust_CBF/data_plot/B{}sample_{}steps_{}_CBF_{}'.format(plant.K,plant.T,plant.obstacle_type,timestr), Sample)
    #     np.save('robust_CBF/data_plot/D{}sample_{}steps_{}_CBF_{}'.format(plant.K,plant.T,plant.obstacle_type,timestr), S_cost)
    # else:
    #     savetxt('robust_CBF/data_plot/A{}sample_{}steps_{}_MPPI_{}.csv'.format(plant.K,plant.T,plant.obstacle_type,timestr), [x_s, y_s], delimiter=',')
    #     savetxt('robust_CBF/data_plot/C{}sample_{}steps_{}_MPPI_{}.csv'.format(plant.K,plant.T,plant.obstacle_type,timestr), [z_s, v_s, w_s], delimiter=',')
    #     np.save('robust_CBF/data_plot/B{}sample_{}steps_{}_MPPI_{}'.format(plant.K,plant.T,plant.obstacle_type,timestr), Sample)
    #     np.save('robust_CBF/data_plot/D{}sample_{}steps_{}_MPPI_{}'.format(plant.K,plant.T,plant.obstacle_type,timestr), S_cost)