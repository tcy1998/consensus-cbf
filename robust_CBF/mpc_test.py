import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def double_integral_system_step(X, U_vx, U_vy):
    # System dynamics: double integrator with control inputs as x and y accleleration
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
    U = np.array([U_vx, U_vy])
    delta_X = np.matmul(A, X) + np.matmul(B, U)
    return X + delta_X * dt

def mpc_cost(U, X, target):
    cost = 0
    # print(U.shape)
    U_vx, U_vy = U.reshape(horizon, 2).T
    for i in range(horizon):
        X = double_integral_system_step(X, U_vx[i], U_vy[i])
        cost += quadratic_cost(X, U_vx[i], U_vy[i], target)
    return cost

def quadratic_cost(x, U_vx, U_vy, target):
    # Example quadratic cost: minimize distance to target position
    position_cost = np.sum((x[:2] - target) ** 2)
    control_cost = U_vx ** 2 + U_vy ** 2
    # print(position_cost, control_cost)
    return position_cost + control_cost

def solve_mpc(x0, target):
    U_max = np.array([1.0, 1.0])
    U_min = np.array([-1.0, -1.0])
    bounds = [(U_min, U_max) for idx in range(horizon)]   # Control action bounds
    x = x0
    u_init_guess = np.zeros((horizon, 2))
    res = minimize(mpc_cost, u_init_guess, args=(x, target))
    if res.success:
        return res.x
    else:
        return None


# Create MPC solver instance
horizon = 10
dt = 0.1

# Set the target position
target_position = np.array([3, 5])

# Solve MPC for initial state x0
initial_position = np.array([0.0, 0.0, 0.0, 0.0])
state_position = initial_position
trajectory = [initial_position]
control = []
episode = 2500
u_init = np.array([0.1, 0.1])
for i in range(episode):
    u_optimal = solve_mpc(initial_position, target_position)
    # print(u_optimal)
    u_vx, u_vy = u_optimal.reshape(horizon, 2).T
    state_position = double_integral_system_step(state_position, u_vx[0], u_vy[0])
    
    trajectory.append(state_position)
    control.append(u_optimal)
print(control)
print("Final position: ", state_position)
plt.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'b-')
plt.plot(target_position[0], target_position[1], 'r*')
plt.show()


