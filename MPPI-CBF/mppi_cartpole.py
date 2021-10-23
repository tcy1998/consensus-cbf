import numpy as np
import math
import matplotlib.pyplot as plt

class MPPI():

    def __init__(self, K, T, U, lamda=0.01, mu=0, sigma=0.5, iter=200):
        self.K = K
        self.T = T
        self.U = U
        self.u_init = 0

        self.lamda = lamda

        self.mu = mu
        self.sigma = sigma
        self.x_init = np.matrix([[0.0],[0.0],[0.0],[0.0]])
        self.iter = iter
        self.Cost_total = np.zeros(shape=(self.K))
        self.X, self.COST = np.zeros(shape=(iter,4)), np.zeros(shape=(iter))
        

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02

        self.noise = np.random.normal(loc=self.mu, scale=self.sigma, size=(self.K, self.T))
        

    def cost_function(self, k):
        x_start = self.x_init.copy()
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            # print(x_start,perturbed_action_t)
            x_start = self.dynamic(x_start,perturbed_action_t)
            penalty = self.cost(x_start) + self.U[t]*perturbed_action_t*0.1
            self.Cost_total[k] += penalty
        term_cost = self.Term_Cost(x_start)
        self.Cost_total[k] += term_cost    


    def cost(self, x):
        return 1*x[0]**2 +5* (x[2]-np.pi)**2  + 0.01*x[1]**2 + 0.1*x[3]**2

    def Term_Cost(self, x):
        return 70*(x[2]-np.pi)**2  + 10*x[1]**2 + 50*x[3]**2

    def dynamic(self, x, u):
        x_state = x[0]
        x_dot_state = x[1]
        theta = x[2]
        theta_dot = x[3]
        force = u
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        # print(theta)
        temp = self.polemass_length * theta_dot ** 2 * sintheta * self.masspole
        thetaacc = (-costheta * temp - force * costheta - self.masspole * self.gravity * sintheta - self.masscart * self.gravity * sintheta)/ ((self.masscart+ self.masspole*sintheta**2)*self.polemass_length)
        xacc = (temp + force + self.masspole * sintheta * costheta * self.gravity) / (self.masscart + self.masspole*sintheta**2)
        x_state = x_state + self.tau * x_dot_state
        x_dot_state = x_dot_state + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        x_new = np.array([x_state,x_dot_state,theta, theta_dot])

        return x_new

    def ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def control(self, iter=1000):
        for _ in range(iter):
            self.noise = np.random.normal(loc=self.mu, scale=self.sigma, size=(self.K, self.T))
            for k in range(self.K):
                self.cost_function(k)

            beta = np.min(self.Cost_total)
            # print(self.Cost_total)
            nonzero_cost = self.ensure_non_zero(cost=self.Cost_total, beta=beta, factor = 1.0/self.lamda)
            # print(nonzero_cost)
            eta = np.sum(nonzero_cost)
            omega = nonzero_cost/eta

            self.U += [np.sum(omega * self.noise[:, t]) for t in range(self.T)]
            # print(self.U[0])
            self.x_init = self.dynamic(self.x_init, self.U[0])
            reward = self.Term_Cost(self.x_init) 
            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.Cost_total[:] = 0
            # print(self.x_init,beta)
            self.X[_] = np.transpose(self.x_init)
            self.COST[_]=reward
            

    def plot_figure(self, iter=1000):
        self.t = np.linspace(0,self.tau*self.iter, num=iter)
        plt.plot(self.t, self.COST)
        plt.show()
        plt.plot(self.t, np.transpose(self.X)[0],label='x')
        plt.plot(self.t, np.transpose(self.X)[1],label='x_dot')
        plt.plot(self.t, np.transpose(self.X)[2],label='theta')
        plt.plot(self.t, np.transpose(self.X)[3],label='theta_dot')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()


if __name__ == "__main__":
    Timesteps = 20
    N_samples = 1000
    Action_low = -20.0
    Action_high = 20.0

    U = np.random.uniform(low=Action_low,high=Action_high, size=Timesteps)
    mppi = MPPI(K=N_samples, T=Timesteps, U=U, iter=400)
    mppi.control(iter=400)
    mppi.plot_figure(iter=400)