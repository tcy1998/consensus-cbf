import numpy as np
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

class MPPI():
    def __init__(self, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=0.1, u_init=1, noise_gaussian=True, downward_start=True, iter=100):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = [u_init,0]
        self.cost_total = np.zeros(shape=(self.K))
        self.tau = 0.05

        self.x_init = [0, 0, np.pi/4]
        self.t_init = 0
        self.target = [4,4,np.pi/4]
        self.v_desire = 1.0 #0.25*np.sqrt(2)
        self.w_desire = 0.0
        self.X, self.Reward = np.zeros(shape=(iter,3)),np.zeros(shape=(iter))
        self.iter = iter
        self.max_speed = 2
        self.max_angle_speed = 2

        self.obstcle_x = 1.5
        self.obstcle_y = 1.0
        self.r =1.0


    def _compute_total_cost(self, k):
        state = [0,0,0]
        state[:] = self.x_init[:]
        use_cbf = True
        m,s = self.noise_mu, self.noise_sigma
        for t in range(self.T):
            
            if use_cbf == True:
                u_upper = m + 3*s+self.U[t]  #The maximum for the sample action
                u_lower = m - 3*s+self.U[t]
                # print(u_upper,state)
                u_upper += self.cbf(state,u_upper).reshape((2,))-self.U[t]
                u_lower += self.cbf(state,u_lower).reshape((2,))-self.U[t]
                m = (u_upper + u_lower)/2
                s = (u_upper - u_lower)/6
                s = np.clip(s,0,None)
                self.noise[k, t] = np.random.normal(loc=m, scale=s)

            self.noise[k, t, 0] = np.clip(self.noise[k, t, 0], -self.max_speed-self.U[t][0], self.max_speed-self.U[t][0])
            self.noise[k, t, 1] = np.clip(self.noise[k, t, 1], -self.max_angle_speed-self.U[t][1], self.max_angle_speed-self.U[t][1])
            perturbed_action_t = self.U[t] + self.noise[k, t]
            state,reward = self.dynamic(state, perturbed_action_t)
            self.cost_total[k] += reward
        self.cost_total[k] += self.terminal_cost(state, perturbed_action_t)


    def terminal_cost(self, state, u):
        x = state[0]
        y = state[1]
        theta = state[2]
        x_des, y_des, theta_des = self.target[0], self.target[1], self.target[2]
        r = (x_des - x) ** 2 + (y_des - y) ** 2 + (0.1-u[0])**2
        return r

    def angle_normalize(self,x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def obstacle(self, x, y):
        dist = np.sqrt((x-self.obstcle_x)**2+(y-self.obstcle_y)**2)
        # dist = 2.0
        # r = 0.5
        if dist < self.r:
            return 1000
        else:
            return 0

    def dynamic(self, state, u):
        # print(state)True
        constraint_use = False
        x = state[0]
        y = state[1]
        theta = state[2]

        x_des, y_des, theta_des = self.target[0], self.target[1], self.target[2]
        u_v = u[0]
        u_w = u[1]      

        x_dot = np.cos(theta) * u_v
        y_dot = np.sin(theta) * u_v
        theta_dot = u_w
        newx = x + self.tau * x_dot
        newy = y + self.tau * y_dot
        newtheta = theta + self.tau * theta_dot

        state = [newx, newy, newtheta]
        costs = (x_des - newx) ** 2 + (y_des - newy) ** 2 + 1*(self.v_desire-u[0])**2# +1* (theta_des - theta)**2 #+ 10*(self.w_desire-u[1])**2

        if constraint_use == True:
            costs += self.obstacle(newx, newy)
            # print(costs)

        return state, costs

    def cbf(self,state,u_norminal):
        position_x, position_y, theta = state[0], state[1], state[2]
        alpha = 1000

        h_x = (position_x - self.obstcle_x)**2 + (position_y - self.obstcle_y)**2 - self.r **2
        g_x = matrix([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]], (3,2))
        partial_hx = matrix([2*(position_x - self.obstcle_x), 2*(position_x - self.obstcle_x), 0], (1,3))
        g = partial_hx * g_x
        h = matrix(alpha * h_x**3) + g * matrix(u_norminal)
        # print(alpha * h_x ** 3,g * matrix(u_norminal))
        P = matrix([[1.0,0.0],[0.0,1.0]])
        q = matrix([0.0,0.0])

        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,-g,h)
        u_cbf = np.array(sol['x'])
        # u_cbf = self.qp_solver(-g,h)
        return u_cbf

    def qp_solver(self,g,h):
        g_1 = g[0]
        g_2 = g[1]
        u_1 = h*g_1/(g_1**2+g_2**2)
        u_2 = h*g_2/(g_1**2+g_2**2)
        return np.array([u_1[0],u_2[0]])

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))


    def control(self, iter=1000):
        
        for _ in range(iter):
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, 2))
            for k in range(self.K):
                self._compute_total_cost(k)
            
            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)
            eta = np.sum(cost_total_non_zero)
            # print(eta)
            omega = (1/eta) * cost_total_non_zero

            D_u = np.zeros((self.T,2))
            # print(np.shape(omega.T), np.shape(self.noise),np.shape(self.U))
            # print(self.cost_total)
            # print(omega)
            for t in range(self.T):
                # print(np.shape(omega),t)
                delta_U = np.matmul(omega.reshape((1,self.K)),self.noise[:, t])
                # delta_U = np.sum(self.noise[:, t].T * omega)
                D_u[t]=delta_U
            self.U += D_u

            print(self.U[0])
            s,r1 = self.dynamic(self.x_init,self.U[0])
            r = self.terminal_cost(s, self.U[0])
            self.U = np.roll(self.U, -1)  # shift all elements to the left          
            self.U[-1] = self.u_init  #
            self.cost_total[:] = 0
            self.x_init = s
            print(self.x_init,r)

            #Save data for plotting
            self.X[_] = np.transpose(self.x_init)
            self.Reward[_] = r

    def plot_figure(self, iter=1000):
        self.t = np.linspace(0,self.tau*self.iter, num=iter)
        plt.plot(self.t, self.Reward)
        plt.show()
        plt.plot(np.transpose(self.X)[0], np.transpose(self.X)[1],label='x-y')
        circle1 = plt.Circle((self.obstcle_x, self.obstcle_y), self.r, color='r', fill=False)
        ax = plt.gca()
        ax.add_artist(circle1)
        plt.show()
        plt.plot(self.t, np.transpose(self.X)[2])
        plt.show()


if __name__ == "__main__":
    TIMESTEPS = 20   # T
    N_SAMPLES = 200  # K
    ACTION_LOW = -10.0
    ACTION_HIGH = 10.0

    noise_mu = 0
    noise_sigma = 5
    lambda_ = 1.0
    iteration = 100

    # U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(TIMESTEPS,2))  # pendulum joint effort in (-2, +2)
    U = np.zeros((TIMESTEPS,2))
    U.T[:1] = 2.5
    # print(U)
    mppi_gym = MPPI(K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, noise_gaussian=True,iter=iteration)
    mppi_gym.control(iter=iteration)
    mppi_gym.plot_figure(iter=iteration)