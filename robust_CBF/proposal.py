from main import *

class MPPI_control_naive:

    def __init__(self, init_u, init_x):
        self.mdl = Unicycle_dynamic()
        self.d = self.mdl.d
        self.m = self.mdl.m
        self.K = self.mdl.K
        self.T = self.mdl.T
        
        # 0. Initial control
        # self.u_ts = torch.Tensor(np.zeros((self.T, self.m)))
        self.u_ts = torch.Tensor(init_u.repeat(self.T).reshape(self.T, self.m))

        # 1.Initial State 
        self.x = torch.Tensor(init_x)

        # 2. Hyper parameter
        self.mu = torch.Tensor(np.zeros(self.m))
        self.sigma = torch.Tensor(np.eye(self.m))

        self.Lambda = 1.0
        self.control_limit = self.mdl.control_limit
        
    def mppi_control(self,x):
        self.x = x.view(self.d)

        # Step 0. Initilize the signals: (1) random explroation noise; (2) sum of cost, (3) state values
        x_init_dK = torch.transpose(self.x.repeat(self.K, 1), 0, 1) # Size (d, K)
        eps_TmK = torch.randn(self.T, self.m, self.K)               # Size (T, m, K)
        S_K = torch.Tensor(np.zeros(self.K))                        # Size (K)
        u_Tm = self.u_ts                                            # Size (T, m)
        x_K = x_init_dK                                             # Size (d, K)
        state_sample = torch.Tensor(np.zeros((self.T, self.d, self.K)))

        for t in range(self.T):
            u_t = u_Tm[t]
            u_K = torch.transpose(u_t.repeat(self.K, 1), 0, 1)
            eps_mK = eps_TmK[t]
            
            # u_K = torch.clamp(u_K + eps_mK, -self.control_limit, self.control_limit)
            u_K = u_K + eps_mK
            
            x_K_next = self.mdl.dynamic(x_K, u_K)
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
        U_T = U_T + torch.sum(eps_TmK*omega_tilde, 2)/eta

        # Step 5. Update the controls
        self.u_ts[:-1] = U_T[1:]
        self.u_ts[-1] = torch.Tensor(np.zeros((self.m)))

        # Step 6. return numpy array
        U_np = self.u_ts.data.numpy()
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

    init_u = np.array([4.0, 0.0])
    init_x = np.array([0.0, 0.0, np.pi/4])

    mppi = MPPI_control_naive(init_u, init_x)
    plant = Unicycle_Environment()
    safe_control = naive_CBF()
    time_steps = 250


    obs = plant.reset()

    x_s, y_s, z_s = [], [], []
    v_s, w_s = [], []
    Sample = []
    S_cost = []

    for t in range(time_steps):
        U_np, X_np, S_K = mppi.mppi_control(obs)
        x, y, z = obs
        dist =(plant.target_pos_x - x)**2 + (plant.target_pos_y - y)**2
        if dist < 0.25**2:
            print(dist, x, y, t)
            break

        u = torch.Tensor(U_np[0])
        obs = plant.step(u)         #Update dynamics

        x_s.append(x.data.numpy())
        y_s.append(y.data.numpy())
        z_s.append(z)
        v_s.append(u[0])
        w_s.append(u[1])
        S_cost.append(S_K)

    mppi.plot(x_s, y_s)