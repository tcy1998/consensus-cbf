import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import time

Theta, Theta_d, Theta_dd = [], [], []
T, dt = 40, 0.001
gamma, alpha = 0.5, np.pi/8
theta, theta_d, theta_dd = alpha, 0.0, 0.0
g, l = 9.8, 1.0

def remless_dynamic(state, u):
    theta, theta_d = state[0], state[1]
    if theta >= gamma + alpha:
        theta_d = theta_d * np.cos(2*alpha)
        theta = theta_d * dt + gamma - alpha
        return [theta, theta_d]
        # Theta.append(theta)
        # Theta_d.append(theta_d)
        # print("hit")
    elif theta <= gamma - alpha:
        theta_d = theta_d * np.cos(2*alpha)
        theta = theta_d * dt + gamma + alpha
        return [theta, theta_d] 
    else:
        theta_d += g/l * np.sin(theta) * dt + u * dt
        theta += theta_d * dt
        return [theta, theta_d]
        # Theta.append(theta)
        # Theta_d.append(theta_d)

def controller(state):
    k1 = -100
    k2 = -100
    # k = 0
    return k1 * (state[0]-0.5) + k2 * state[1]

def func_h_bar(state):
    return 0.01-state[1]**2

def partial_h_bar(state):
    # print(-2*state[1])
    return matrix([[0], [-2*state[1]]])

def QP_CBF(u_norminal, state):
    alpha = 1000
    P = matrix([1.0])
    q = matrix([0.0])

    g_x = matrix([[0],[1]])
    
    g = partial_h_bar(state)*g_x.T
    hx = alpha * (func_h_bar(state))**1 + g[0]*u_norminal
    G = -g
    h = matrix([matrix(hx)]) 
    # print(P,q,G,h)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P,q,G,h)
    u_cbf = np.array(sol['x'])
    # print(u_cbf)
    return u_cbf[0]

use_cbf = True

start_time = time.time()
for t in range(int(T/dt)):
    u = controller([theta, theta_d])
    if theta_d < 0.1:
        use_cbf = False
    else:
        use_cbf = True
    if use_cbf == True:
        u += QP_CBF(u, [theta, theta_d])[0]
    [theta, theta_d] = remless_dynamic([theta,theta_d], u)
    Theta.append(theta)
    Theta_d.append(theta_d)
    t += dt

print("--- %s seconds ---" % (time.time() - start_time))

t_linspace = np.linspace(0, T, num=int(T/dt))
plt.plot(Theta, Theta_d)
plt.axvline(gamma+alpha)
plt.axvline(gamma-alpha)
plt.show()

plt.plot(t_linspace, Theta)
plt.plot(t_linspace, Theta_d)
plt.show()