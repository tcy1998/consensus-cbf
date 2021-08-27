import numpy as np
import math

class CBF():
    def __init__(self, T):
        self.T = T
        self.safe_distance = 0.5

    def dynamic(self, x, y, theta, u_v, u_w):
        x_dot = np.cos(theta) * u_v
        y_dot = np.sin(theta) * u_v
        theta_dot = u_w
        newx = x + self.tau * x_dot
        newy = y + self.tau * y_dot
        newtheta = theta + self.tau * theta_dot

        state = [newx, newy, newtheta]

        return state

    def c_(self, x_i, x_j, y_i, y_j):
        return (x_i - x_j) **2 + (y_i -y_j) ** 2 - self.safe_distance **2
 
    def leader_control(self, state, target, k, lambda_, obstacle):
        x, y, theta = state[0], state[1], state[2]
        x_id, y_id = target[0], target[1]
        

    def follower_control(self, state, target, k, lambda_, obstacle):
        x, y, theta = state[0], state[1], state[2]

    
if __name__ == "__main__":
    TIMESTEPS = 10   # T
    N_SAMPLES = 2000  # K
    ACTION_LOW = -10.0
    ACTION_HIGH = 10.0