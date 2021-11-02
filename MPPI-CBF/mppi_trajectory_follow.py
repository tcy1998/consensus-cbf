import numpy as np
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import cvxpy as cp
import numpy as np
import cvxopt
import time
from numpy import savetxt


class MPPI():
    def __init__(self, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=0.1, u_init=2.5, noise_gaussian=True, downward_start=True, iter=100):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = [u_init,0]
        self.cost_total = np.zeros(shape=(self.K))
        self.tau = 0.05


        # target parameters
        self.x_init = [0, 0, np.pi/4]
        self.t_init = 0
        self.target = [4,4,np.pi/4]
        self.v_desire = 0.04*np.sqrt(2)
        # self.w_desire = 0.0
        self.X, self.Reward = np.zeros(shape=(iter,3)),np.zeros(shape=(iter))
        self.iter = iter
        self.max_speed = 5.0
        self.max_angle_speed = 5.0


        # Obstacle parameters
        self.obstcle_x = 2.2
        self.obstcle_y = 2.0
        self.r =0.5

        self.Obstacle_X = matrix([1.5, 1.5, 3.5])
        self.Obstacle_Y = matrix([1.0, 3.3, 2.5])
        self.R = matrix([1.0, 1.0, 1.0])

        self.use_cbf = False            #use cbf
        self.constraint_use = False      #Orignal MPPI
        self.multi_ = False             #multi obstacles
        self.naive_cbf = True

        self.plot_sample = True         #plot sample trajectory
        self.sample_data = np.zeros((self.K,self.T,3))
        self.plot_sample_time = 7

        self.save_data = False


    def _compute_total_cost(self, k, time_step):
        state = [0,0,0]
        state[:] = self.x_init[:]
        m,s = self.noise_mu, self.noise_sigma
        for t in range(self.T):
            # print(state, t)
            if time_step == self.plot_sample_time:
                self.sample_data[k, t, :] = state
            # if using cbf update the mean and variance
            # print(m,s)
            if self.use_cbf == True:
                m_,s_ = self.sdp_cbf(state,self.U[t], m,s)  #Using function to return new safe and mean variance
                # print(m_, s_)
                m = m+m_.reshape(1,2)
                s += s_
                # print(m,s)
                # print(m[0], s, np.random.multivariate_normal(m[0], s))
                self.noise[k, t, :] = np.random.multivariate_normal(m[0], s)    #m[0] is because will become a nested array
            
            self.noise[k, t, 0] = np.clip(self.noise[k, t, 0], -self.max_speed-self.U[t][0], self.max_speed-self.U[t][0])   # Speed limit
            self.noise[k, t, 1] = np.clip(self.noise[k, t, 1], -self.max_angle_speed-self.U[t][1], self.max_angle_speed-self.U[t][1])   #Angular speed limit
            perturbed_action_t = self.U[t] + self.noise[k, t]
            state,reward = self.dynamic(state, perturbed_action_t)          #dynamic updates
            
            self.cost_total[k] += reward + self.U[t] @ np.linalg.inv(s) @ self.noise[k,t].reshape(2,1)          #Calculating the reward
        self.cost_total[k] += self.terminal_cost(state, perturbed_action_t)

    # Terminal cost
    def terminal_cost(self, state, u):
        x = state[0]
        y = state[1]
        theta = state[2]
        x_des, y_des, theta_des = self.target[0], self.target[1], self.target[2]
        dist = (x - x_des)**2 + (y-y_des)**2
        r = 0
        r += dist
        return r

    # One obstacle reward
    def obstacle(self, x, y):
        dist = np.sqrt((x-self.obstcle_x)**2+(y-self.obstcle_y)**2)

        if dist < self.r:
            return 10000
        else:
            return 0

    # Multi obstacle reward
    def multi_obstacle(self, x, y):
        r = 0
        for i in range(len(self.Obstacle_X)):
            dist = np.sqrt((x-self.Obstacle_X[i])**2+(y-self.Obstacle_Y[i])**2)
            if dist < self.R[i]:
                r += 10000
        return r

    # System dynamic of unicycle, return new states and the costs
    def dynamic(self, state, u):
        
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
        costs = 10*(x_des - newx) ** 2 + 10* (y_des - newy) ** 2 + 1*(self.v_desire-u[0])**2

        if self.constraint_use == True:
            costs += self.obstacle(newx, newy)
        if self.multi_ == True:
            costs += self.multi_obstacle(newx, newy)
            # print(costs)

        return state, costs

    def cbf(self,state,u_norminal):
        position_x, position_y, theta = state[0], state[1], state[2]
        alpha = 1000

        h_x = (position_x - self.obstcle_x)**2 + (position_y - self.obstcle_y)**2 - self.r **2
        g_x = matrix([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]], (3,2))
        partial_hx = matrix([2*(position_x - self.obstcle_x), 2*(position_y - self.obstcle_y), 0], (1,3))
        g = partial_hx * g_x
        h = matrix(alpha * h_x**3) + g * matrix(u_norminal)

        P = matrix([[1.0,0.0],[0.0,1.0]])
        q = matrix([0.0,0.0])

        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,-g,h)
        u_cbf = np.array(sol['x'])
        return u_cbf

    def sdp_cbf(self, state, u_norminal, mu, sigma):
        position_x, position_y, theta = state[0], state[1], state[2]
        alpha = 100

        A_ = np.zeros((len(self.R),2))
        B = np.zeros(len(self.R))

        if self.multi_ == True:
            for i in range(len(self.R)):
                h_x = (position_x - self.Obstacle_X[i])**2 + (position_y - self.Obstacle_Y[i])**2 - self.R[i] **2
                g_x = matrix([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]], (3,2))
                partial_hx = matrix([2*(position_x - self.Obstacle_X[i]), 2*(position_y - self.Obstacle_Y[i]), 0.0], (1,3))
                A_[i] = partial_hx * g_x
                C = matrix(A_[i], (2,1))
                B[i] = -matrix(alpha * h_x**3) - A_[i] @ matrix(u_norminal)
            variance = cp.Variable((2, 2), PSD=True)
            mean = cp.Variable((2, 1))
            constraints = [matrix(A_[0],(1,2)) @ variance @ matrix(A_[0],(2,1))+ A_[0] @ mean >> B[0],\
                matrix(A_[1],(1,2)) @ variance @ matrix(A_[1],(2,1))+ A_[1] @ mean >> B[1],\
                matrix(A_[2],(1,2)) @ variance @ matrix(A_[2],(2,1))+ A_[2] @ mean >> B[2],
                variance >> 0]
            objective = cp.Minimize(cp.trace(variance)+cp.norm(mean,2))
            prob = cp.Problem(objective, constraints)
            prob.solve()
        else:

            h_x = (position_x - self.obstcle_x)**2 + (position_y - self.obstcle_y)**2 - self.r **2
            g_x = matrix([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]], (3,2))
            partial_hx = matrix([2*(position_x - self.obstcle_x), 2*(position_y - self.obstcle_y), 0], (1,3))
            A = partial_hx * g_x
            b = -matrix(alpha * h_x**3) - A * matrix(u_norminal)

            variance = cp.Variable((2, 2), PSD=True)
            mean = cp.Variable((2, 1))
            # print(A, A.T)
            constraints = [A @ variance @ A.T+ A @ mean >> b, variance >> 0]
            objective = cp.Minimize(cp.trace(variance)+cp.norm(mean,1))
            # start_time = time.time()
            prob = cp.Problem(objective, constraints)
            prob.solve()

            # print("The optimal value is", prob.value)
            # print("A solution mean is")
            # print(mean.value)
            # print("A solution variance is")
            # print(variance.value)
            # print("--- %s seconds ---" % (time.time() - start_time))

        return mean.value, variance.value


    # Self writing qp solver only usable for simple QP problem
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
            self.noise = np.random.multivariate_normal(self.noise_mu, self.noise_sigma, size=(self.K, self.T))
            for k in range(self.K):
                self._compute_total_cost(k, _)
            
            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)
            eta = np.sum(cost_total_non_zero)
            # print(eta)
            omega = (1/eta) * cost_total_non_zero

            D_u = np.zeros((self.T,2))
            for t in range(self.T):
                # print(np.shape(omega),t)
                delta_U = np.matmul(omega.reshape((1,self.K)),self.noise[:, t])
                D_u[t]=delta_U
            self.U += D_u

            if self.naive_cbf == True:
                # print(self.U[0], self.cbf(self.x_init,self.U[0]).reshape(1,2))
                self.U[0] += self.cbf(self.x_init,self.U[0]).T[0]

            s,r1 = self.dynamic(self.x_init,self.U[0])
            r = self.terminal_cost(s, self.U[0])
            self.U = np.roll(self.U, -2)  # shift all elements to the left          
            self.U[-1] = self.u_init  #
            self.cost_total[:] = 0
            self.x_init = s
            print(self.x_init,r)

            #Save data for plotting
            self.X[_] = np.transpose(self.x_init)
            self.Reward[_] = r

            #if the distance is close to the target jump out the loop
            if ((self.x_init[0]-self.target[0])**2 + (self.x_init[1]-self.target[1])**2 < 0.04):
                self.iter = _
                break

    # Plot the figures
    def plot_figure(self, iter=1000):
        self.t = np.linspace(0,self.tau*self.iter, num=self.iter)
        print(self.iter)
        plt.plot(self.t, self.Reward[0:self.iter])
        plt.show()
        self.X = self.X[0:self.iter]
        plt.plot(np.transpose(self.X)[0], np.transpose(self.X)[1],label='x-y')
        target_circle = plt.Circle((self.target[0], self.target[1]), 0.2, color='b', fill=False)
        ax = plt.gca()
        ax.add_artist(target_circle)
        
        if self.multi_ == True:
            for i in range(len(self.R)):
                circle1 = plt.Circle((self.Obstacle_X[i], self.Obstacle_Y[i]), self.R[i], color='r', fill=False)
                ax = plt.gca()
                ax.add_artist(circle1)
        else:
            circle1 = plt.Circle((self.obstcle_x, self.obstcle_y), self.r, color='r', fill=False)
            ax = plt.gca()
            ax.add_artist(circle1)
        plt.xlim([0,5])
        plt.ylim([0,5])
        plt.show()
        plt.plot(self.t, np.transpose(self.X)[2])
        plt.show()

        if self.plot_sample == True:
            if self.multi_ == False:
                for i in range(len(self.sample_data)):
                    plt.plot(np.transpose(self.sample_data[i])[0], np.transpose(self.sample_data[i])[1], color='g')
                circle1 = plt.Circle((self.obstcle_x, self.obstcle_y), self.r, color='r', fill=False)
                ax = plt.gca()
                ax.add_artist(circle1)
                plt.xlim([1,4])
                plt.ylim([1,4])
                plt.show()
            else:
                for i in range(len(self.sample_data)):
                    plt.plot(np.transpose(self.sample_data[i])[0], np.transpose(self.sample_data[i])[1], color='g')
                for i in range(len(self.R)):
                    circle1 = plt.Circle((self.Obstacle_X[i], self.Obstacle_Y[i]), self.R[i], color='r', fill=False)
                    ax = plt.gca()
                    ax.add_artist(circle1)
                plt.xlim([0,4])
                plt.ylim([0,4])
                plt.show()

        self.sample_data_reshape = self.sample_data.reshape(self.sample_data.shape[0], -1)
        print(np.shape(self.sample_data), np.shape(self.sample_data_reshape))

        if self.save_data == True:
            if self.use_cbf == True:
                if self.multi_ == True:
                    savetxt('{}sample_{}steps_multi_CBF.csv'.format(self.K,self.T), self.X, delimiter=',')
                    savetxt('{}sample_{}steps_multi_CBF_cost.csv'.format(self.K,self.T), self.Reward, delimiter=',')
                    savetxt('{}sample_{}steps_multi_CBF_{}timestep.csv'.format(self.K,self.T,self.plot_sample_time), self.sample_data_reshape, delimiter=',')
                else:
                    savetxt('{}sample_{}steps_single_CBF.csv'.format(self.K,self.T), self.X, delimiter=',')
                    savetxt('{}sample_{}steps_single_CBF_cost.csv'.format(self.K,self.T), self.Reward, delimiter=',')
                    savetxt('{}sample_{}steps_single_CBF_{}timestep.csv'.format(self.K,self.T,self.plot_sample_time), self.sample_data_reshape, delimiter=',')
            else:
                if self.multi_ ==True:
                    savetxt('{}sample_{}steps_multi_MPPI.csv'.format(self.K,self.T), self.X, delimiter=',')
                    savetxt('{}sample_{}steps_multi_MPPI_cost.csv'.format(self.K,self.T), self.Reward, delimiter=',')
                    savetxt('{}sample_{}steps_multi_MPPI_{}timestep.csv'.format(self.K,self.T,self.plot_sample_time), self.sample_data_reshape, delimiter=',')
                else:
                    savetxt('{}sample_{}steps_single_MPPI.csv'.format(self.K,self.T), self.X, delimiter=',')
                    savetxt('{}sample_{}steps_single_MPPI_cost.csv'.format(self.K,self.T), self.Reward, delimiter=',')
                    savetxt('{}sample_{}steps_single_MPPI_{}timestep.csv'.format(self.K,self.T,self.plot_sample_time), self.sample_data_reshape, delimiter=',')


if __name__ == "__main__":
    TIMESTEPS = 20  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -10.0
    ACTION_HIGH = 10.0

    noise_mu = (0, 0)
    noise_sigma = [[1, 0], [0, 0.1]]
    lambda_ = 1.0
    iteration = 100

    U = np.zeros((TIMESTEPS,2))
    U.T[:1] = 2.5
    # print(U)
    mppi_unicycle = MPPI(K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=2.5, noise_gaussian=True,iter=iteration)
    mppi_unicycle.control(iter=iteration)
    mppi_unicycle.plot_figure(iter=iteration)