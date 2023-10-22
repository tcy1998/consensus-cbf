import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from tqdm import tqdm

class Unicycle_CBF_multi_agent_circle_targets():
    # x: state
    # u: control input
    # return: CBF value
    def __init__(self, num_agents):

        self.R = 10
        self.old_e = np.zeros((num_agents, 2))
        self.e_integral = np.zeros((num_agents, 2))
        self.num_agents = num_agents
        self.dt = 0.02
        self.robot_radius = 0.5
        self.theta_max = np.pi/4

    def cbf_solver(self, obstacle_xs, state_x, state_y, state_theta, uv_n):
        gx = matrix([[np.cos(state_theta), np.sin(state_theta), 0], [0, 0, 1]])         # gx 3*2 marix

        Hx, Gx = [], []
        for obstacle_x in obstacle_xs:
            middle_value = np.cos(state_theta)*(state_x - obstacle_x[0]) + np.sin(state_theta)*(state_y - obstacle_x[1])
            ggx = [-middle_value, 0]
            Gx.append(ggx)
            hx = np.linalg.norm([state_x, state_y] - obstacle_x[:2])**2  + middle_value * uv_n                     # hx scalar, only times uv because ggx[1] = 0
            Hx.append(hx)

        P = matrix(np.eye(2))                                                         # P 2*2 matrix    
        q = matrix(np.zeros((2, 1)))                                                  # q 2*1 matrix


        G = matrix(Gx).T
        h = matrix(Hx)                  # G * u <= h so the previous value ggx is negative and hx is positive

        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,G,h)
        u_cbf = np.array(sol['x'])
        return u_cbf
    
    # def alpha_function(self, x):
    #     alpha = 1e-1
    #     return alpha*x**3
    
    def alpha_function(self, x):
        alpha = 1e2
        return alpha * (np.exp(x)/ (np.exp(x) + 1) - 0.5)
    
    def single_agent_cbf_solver(self, obstacle_xs, state_x, state_y, state_theta, uv_n):
        Hx, Gx = [], []
        Lghx = [np.cos(state_theta)*(state_x - obstacle_xs[0]) + np.sin(state_theta)*(state_y - obstacle_xs[1]), 0]
        Lghx = np.array(Lghx).reshape(1,2)
        hx = (state_x - obstacle_xs[0])**2 + (state_y - obstacle_xs[1])**2 - self.robot_radius ** 2
        hhx = self.alpha_function(hx) + Lghx @ uv_n.T

        P = matrix(np.eye(2))                                                         # P 2*2 matrix    
        q = matrix(np.zeros((2, 1)))                                                  # q 2*1 matrix

        ### -Lghx * (delta_u + uv_n) <= alpha_hx ###
        G = matrix(-Lghx)
        h = matrix(hhx)                  # G * u <= h so the previous value Lghx is negative and hhx is positive

        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,G,h)
        u_cbf = np.array(sol['x'])
        return u_cbf

    
    def PD_controller(self, state_x, target, agent_index):
        dx = target[0] - state_x[0]
        dy = target[1] - state_x[1]
        theta = np.arctan2(dy, dx)
        dtheta = theta - state_x[2]

        ep1, ep2 = np.sqrt(dx**2 + dy**2), dtheta
        ed1 = ep1 - self.old_e[agent_index][0]
        ed2 = ep2 - self.old_e[agent_index][1]
        self.old_e[agent_index][0] = ep1
        self.old_e[agent_index][1] = ep2

        # kpv, kpw = -.3, 0.005 
        # kdv, kdw = -1.0, 1.4

        kpv, kpw = -.5, -0.01 
        kdv, kdw = 0.2, 1.0
        v = kpv*ep1 + kdv*ed1
        w = kpw*ep2 + kdw*ed2

        return v, w
    
    def PID_controller(self, state_x, target, agent_index):
        dx = target[0] - state_x[0]
        dy = target[1] - state_x[1]
        theta = np.arctan2(dy, dx)
        dtheta = theta - state_x[2]

        ep1, ep2 = np.sqrt(dx**2 + dy**2), dtheta
        ed1 = ep1 - self.old_e[agent_index][0]
        ed2 = ep2 - self.old_e[agent_index][1]
        self.old_e[agent_index][0] = ep1
        self.old_e[agent_index][1] = ep2
        ei1 = self.e_integral[agent_index][0] + ep1
        ei2 = self.e_integral[agent_index][1] + ep2
        self.e_integral[agent_index][0] = ei1
        self.e_integral[agent_index][1] = ei2

        kpv, kpw = -.5, -0.01
        kdv, kdw = 0.2, 1.0
        kiv, kiw = -0.0001, -1e-6

        v = kpv*ep1 + kdv*ed1 + kiv*ei1
        w = kpw*ep2 + kdw*ed2 + kiw*ei2

        return v, w

    
    # def UN_LN_controller(self, virtual_target, true_position, error):
    #     k1 = -40
    #     k2 = -100
    #     k3 = -2                                        # A Linear Controller for the Unicycle Model
    #     desire_speed = 5*np.pi/50
    #     desire_angular = 2*np.pi/50
    #     true_x_error = np.cos(true_position[2][0])*error[0][0] + np.sin(true_position[2][0])*error[1][0]
    #     true_y_error = np.cos(true_position[2][0])*error[1][0] - np.sin(true_position[2][0])*error[0][0]

    #     u_v = -k1 * true_x_error
    #     u_r = -k2 * np.sign(desire_speed) * true_y_error - k3 * error[2][0]

    #     u_true = np.array([[u_v], [u_r]])
    #     return u_true
    
    # def UN_controller(self, virtual_target, true_position, error,desire_speed = 5*np.pi/50,desire_angular = 2*np.pi/50):
    #     k2 = -10000                                      # A Nonlinear Controller for the Unicycle Model 
    #     true_x_error = np.cos(true_position[2][0])*error[0][0] + np.sin(true_position[2][0])*error[1][0]
    #     true_y_error = np.cos(true_position[2][0])*error[1][0] - np.sin(true_position[2][0])*error[0][0]

    #     u_v = -self.k_function(desire_speed, desire_angular) * true_x_error
    #     u_r = -k2 * np.sin(error[2][0]) * true_y_error / (error[2][0]+0.0001) - self.k_function(desire_speed, desire_angular) * error[2][0]

    #     u_true = np.array([[u_v], [u_r]])
    #     return u_true
    
    # def k_function(self, x1, x2):
    #     zeta, beta = -100, 100
    #     return 2*zeta*np.sqrt(x1**2+beta*x2**2)

    def update_state(self, state_x, u):
        x = state_x[0] + u[0][0]*np.cos(state_x[2])*self.dt
        y = state_x[1] + u[0][0]*np.sin(state_x[2])*self.dt
        theta = state_x[2] + u[0][1]*self.dt
        return np.array([x, y, theta])

    def main(self):
        ### Define the initial state and target state ###
        Init_state, Target_state = [], []
        for i in range(self.num_agents):
            delta_angle = 2*np.pi/self.num_agents * i
            initial_state = np.array([np.cos(delta_angle) * self.R, np.sin(delta_angle) * self.R, delta_angle])
            target_state = np.array([np.cos(delta_angle+np.pi) * self.R, np.sin(delta_angle+np.pi) * self.R, delta_angle])
            Init_state.append(initial_state)
            Target_state.append(target_state)
        State = Init_state.copy()

        ### Define the obstacle position ###

        ### Define the control input ###
        Time_step = 1000
        State_log = []
        State_log_1 = []
        stop_indicator = np.zeros(self.num_agents)
        for t in tqdm(range(Time_step)):
            for i in range(self.num_agents):
                
                v, w = self.PD_controller(State[i], Target_state[i], i)
                u = np.array([v, w]).reshape(1,2)
                obstacle_list = State.copy()[:i] + State.copy()[i+1:]
                # u_cbf = self.cbf_solver(obstacle_list, Init_state[i][0], Init_state[i][1], Init_state[i][2], v)
                # u = u_cbf.reshape(1,2) + np.array([v, w])
                
                ### Update the state of each agent ###
                State[i] = self.update_state(State[i], u)
                if np.linalg.norm(State[i][:2] - Target_state[i][:2]) < 0.1:
                    stop_indicator[i] = 1
            if np.sum(stop_indicator) == self.num_agents:
                break
            
            ### Update the states simultaneously ###
            State = State.copy()
            State_log.append(State)
            State_log_1.append(State[0])

        ### Plot the result ###
        State_log = np.array(State_log)
        State_log_1 = np.array(State_log_1)
        plt.figure()
        for i in range(self.num_agents):
            plt.plot(State_log[:, i, 0], State_log[:, i, 1])
            plt.plot(Target_state[i][0], Target_state[i][1], 'r*')
            plt.plot(Init_state[i][0], Init_state[i][1], 'b*')
        # plt.plot(State_log_1[:, 0], State_log_1[:, 1], 'r')
        plt.show()

    def single_agent_main(self):
        Init_state, Target_state = [-10.0, 0.0, -np.pi], [10.0, 0.0, -np.pi]
        Time_step = 1000
        State_log = [Init_state.copy()]
        State = Init_state.copy()
        U_log, U_cbf_log = [], []

        Use_CBF = False
        obstacle_list = [0, 0]

        for t in tqdm(range(Time_step)):
            # v, w = self.PD_controller(State, Target_state, 0)
            v, w = self.PID_controller(State, Target_state, 0)
            u = np.array([v, w]).reshape(1,2)
            if Use_CBF:
                u_cbf = self.single_agent_cbf_solver(obstacle_list, State[0], State[1], State[2], u)
                u = u_cbf.reshape(1,2) + np.array([v, w])
                U_cbf_log.append(u_cbf.reshape(1,2))
            State = self.update_state(State, u)
            State_log.append(State)
            U_log.append(u)
            if np.linalg.norm(State[:2] - Target_state[:2]) < 0.1:
                break
        plt.figure()
        State_log = np.array(State_log)
        plt.plot(State_log.T[0], State_log.T[1])
        plt.plot(Target_state[0], Target_state[1], 'r*')
        plt.plot(Init_state[0], Init_state[1], 'b.')
        plt.axis('equal')


        ### Plot for circle obstacle ###
        target_circle1 = plt.Circle((obstacle_list[0], obstacle_list[1]), self.robot_radius, color='b', fill=False)
        ax = plt.gca()
        ax.add_patch(target_circle1)

        plt.show()

        U_log = np.array(U_log)
        plt.figure()
        t = np.arange(len(U_log))
        print(U_log.shape)
        plt.plot(t, U_log.T[0].T, label='v')
        plt.plot(t, U_log.T[1].T, label='w')
        plt.legend()
        plt.show()

        if Use_CBF:
            U_cbf_log = np.array(U_cbf_log)
            plt.figure()
            t = np.arange(len(U_cbf_log))
            print(U_cbf_log.shape)
            plt.plot(t, U_cbf_log.T[0].T, label='v')
            plt.plot(t, U_cbf_log.T[1].T, label='w')
            plt.show()

if __name__ == '__main__':
    num_agents = 6
    Unicycle_CBF_multi_agent_circle_targets(num_agents).main()              
    # Unicycle_CBF_multi_agent_circle_targets(num_agents).single_agent_main()        
