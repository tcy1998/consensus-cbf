import numpy as np
import matplotlib.pyplot as plt


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

# def cbf(state, u):



for t in range(int(T/dt)):
    u = controller([theta, theta_d])
    [theta, theta_d] = remless_dynamic([theta,theta_d], u)
    Theta.append(theta)
    Theta_d.append(theta_d)
    t += dt

t_linspace = np.linspace(0, T, num=int(T/dt))
plt.plot(Theta, Theta_d)
plt.axvline(gamma+alpha)
plt.axvline(gamma-alpha)
plt.show()

plt.plot(t_linspace, Theta)
plt.plot(t_linspace, Theta_d)
plt.show()