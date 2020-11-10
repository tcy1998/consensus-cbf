import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


class MPPI():
    def __init__(self, K, T, U, lambda_=1.0, noise_mu=0.0, noise_sigma=0.00001, u_init=1, noise_gaussian=True, downward_start=True, iter=100):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))

        self.max_speed = 8
        self.max_torque = 20.
        self.dt = .05
        self.g = 9.8
        self.m = 1.
        self.l = 1.
        self.tau = 0.02
        self.iter = iter
        self.X, self.Reward = np.zeros(shape=(iter,2)),np.zeros(shape=(iter))

        self.x_init = [np.pi, 1]
        self.theta_dot_bound = 4.0

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)

    def _compute_total_cost(self, k):
        state = [0,0]
        state[:] = self.x_init[:]
        use_cbf = True
        m,s = self.noise_mu, self.noise_sigma
        for t in range(self.T):
            if use_cbf == True:
                u_upper = m + 3*s+self.U[t]  #The maximum for the sample action
                u_lower = m - 3*s+self.U[t]  
                u_upper += self.cbf(state,u_upper)-self.U[t]
                # print(state, u_upper)
                u_lower += self.cbf(state,u_lower)-self.U[t]
                # print(state,u_lower)
                m = (u_upper + u_lower)/2
                s = (u_upper - u_lower)/6
                if s < 0:
                    s=1e-4
                self.noise[k, t] = np.random.normal(loc=m, scale=s)
            perturbed_action_t = self.U[t] + self.noise[k, t]
            state,reward = self.dynamic(state, perturbed_action_t)
            # print(state)
            self.cost_total[k] += reward
            
    def angle_normalize(self,x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def dynamic(self, state, u):
        # print(state)
        use_noise = True
        th = state[0]
        thdot = state[1]
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        # print(u)
        u = np.clip(u, -self.max_torque, self.max_torque)
        costs = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        if use_noise == True:
            noise = np.random.normal(loc=0.0, scale=0.01)
        else:
            noise = 0

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt + noise
        newth = th + newthdot * dt +noise
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        state = [newth, newthdot]

        return state, costs

    def cbf(self, state, u):
        th = state[0]
        th_dot = state[1]
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        alpha = 1
        h = th_dot * 3 * g / l * np.sin(th + np.pi) +alpha * (self.theta_dot_bound**2 - th_dot**2) - 6 * th_dot * u / (m * l ** 2) 
        g = 6 * th_dot / (m * l ** 2)
        P = matrix([1.0])
        Q = matrix([0.0])
        G = matrix([g])
        H = matrix([h])
        solvers.options['show_progress'] = False
        sol = solvers.qp(P,Q,G,H)
        u_cbf = np.array(sol['x'])
        return u_cbf[0][0]

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))


    def control(self, iter=1000):       
        for _ in range(iter):
            use_cbf = False
            if use_cbf == True:
                u_upper = self.noise_mu + 3*self.noise_sigma+self.U[0]  #The maximum for the sample action
                u_lower = self.noise_mu - 3*self.noise_sigma+self.U[0]  
                print(u_upper,u_lower)
                u_upper += self.cbf(self.x_init,u_upper)-self.U[0]
                u_lower += self.cbf(self.x_init,u_lower)-self.U[0]
                self.noise_mu = (u_upper + u_lower)/2
                self.noise_sigma = (u_upper - u_lower)/6
                print(u_upper,u_lower)
                # print(self.noise_mu, self.noise_sigma)
                # print(self.dynamic(self.x_init,u_upper+self.U[0]))

                # u_mean = self.noise_mu + self.U[0]

                self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T))
            for k in range(self.K):
                self._compute_total_cost(k)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)
            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero
            
            
            self.U += [np.sum(omega * self.noise[:, t]) for t in range(self.T)]
            u_input = self.U[0]
            if use_cbf == True:
                u_input += self.cbf(self.x_init, u_input)
            s, r = self.dynamic(self.x_init,u_input)
            print(r)

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.cost_total[:] = 0
            self.x_init = s
            self.X[_] = np.transpose(self.x_init)
            self.Reward[_] = r

    def plot_figure(self, iter=1000):
        self.t = np.linspace(0,self.tau*self.iter, num=iter)
        plt.plot(self.t, self.Reward)
        plt.show()
        plt.plot(self.t, np.transpose(self.X)[0],label='theta')
        plt.plot(self.t, np.transpose(self.X)[1],label='theta_dot')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.axhline(y=self.theta_dot_bound, color='r', linestyle='-')
        plt.axhline(y=-self.theta_dot_bound, color='r', linestyle='-')
        plt.show()

if __name__ == "__main__":
    TIMESTEPS = 10  # T
    N_SAMPLES = 1000  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    noise_mu = 0
    noise_sigma = 10
    lambda_ = 0.0001

    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=TIMESTEPS)  # pendulum joint effort in (-2, +2)

    mppi_gym = MPPI(K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, noise_gaussian=True, iter=200)
    mppi_gym.control(iter=200)
    mppi_gym.plot_figure(iter=200)