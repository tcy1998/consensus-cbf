import numpy as np
import math


class MPPI():
    def __init__(self, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=0.01, u_init=1, noise_gaussian=True, downward_start=True):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))
        self.tau = 0.01

        self.x_init = [0, 0, 0]
        self.t_init = 0

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, 2))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)

    def _compute_total_cost(self, k):
        state = [0,0,0]
        state[:] = self.x_init[:]
        t_new = self.t_init
        # print(self.t_init)
        for t in range(self.T):
            # print(self.noise[k,t])
            perturbed_action_t = self.U[t] + self.noise[k, t]
            
            state,reward = self.dynamic(state, perturbed_action_t, t_new)
            # print(reward)
            self.cost_total[k] += reward
            t_new = t_new + self.tau
            # print(t_new)
        self.cost_total[k] += self.terminal_cost(state, t_new)

    def des_trajectory(self, t):
        return [5-5*math.cos(t), -5*math.sin(t),t]

    def terminal_cost(self, state, t):
        x = state[0]
        y = state[1]
        theta = state[2]
        [x_des, y_des, theta_des] = self.des_trajectory(t)
        r = 1000*(x_des - x) ** 2 + 1000*(y_des - y) ** 2 + (theta_des - theta) ** 2
        return r

    def dynamic(self, state, u, t):
        # print(state)
        x = state[0]
        y = state[1]
        theta = state[2]

        [x_des, y_des, theta_des] = self.des_trajectory(t)
        
        # print(u)
        costs =  10*(x_des - x) ** 2 + 10*(y_des - y) ** 2 + (theta_des - theta) ** 2

        x_dot = math.cos(theta) * u[0]
        y_dot = math.sin(theta) * u[1]
        theta_dot = u[1]
        newx = x + self.tau * x_dot
        newy = y + self.tau * y_dot
        newtheta = theta + self.tau * theta_dot

        state = [newx, newy, newtheta]

        return state, costs



    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))


    def control(self, iter=1000):
        for _ in range(iter):
            for k in range(self.K):
                self._compute_total_cost(k)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)
            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero
            D_u = np.zeros((20,2))
            # print(np.shape(omega.T), np.shape(self.noise),np.shape(self.U))
            for t in range(self.T):
                delta_U = omega.reshape((1,1000)) @ self.noise[:, t]
                D_u[t]=delta_U
            self.U += D_u
            
            s, r = self.dynamic(self.x_init,self.U[0], self.t_init)
            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.cost_total[:] = 0
            self.x_init = s
            self.t_init += self.tau
            print(self.t_init,r)
            print(self.x_init,self.des_trajectory(self.t_init))


if __name__ == "__main__":
    TIMESTEPS = 20  # T
    N_SAMPLES = 1000  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    noise_mu = 0
    noise_sigma = 10
    lambda_ = 0.0001

    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(TIMESTEPS,2))  # pendulum joint effort in (-2, +2)

    mppi_gym = MPPI(K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, noise_gaussian=True)
    mppi_gym.control(iter=1000)