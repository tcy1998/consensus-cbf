import numpy as np
import matplotlib.pyplot as plt


Theta, Theta_d, Theta_dd = [], [], []
T, dt = 1, 0.002
gamma, alpha = 0.5, np.pi/8
theta, theta_d, theta_dd = alpha, 0.0, 0.0
g, l = 9.8, 1.0

def controller(state):
    k= 1
    return -k*state[0]

for t in range(int(T/dt)):
    if theta >= gamma + alpha:
        theta_d = theta_d * np.cos(2*alpha)
        theta = theta_d * dt + gamma -alpha
        Theta.append(theta)
        Theta_d.append(theta_d)
        # print("hit")
    else:
        theta_d += g/l * np.sin(theta) * dt + controller([theta, theta_d])
        theta += theta_d * dt
        Theta.append(theta)
        Theta_d.append(theta_d)
    t += dt

t_linspace = np.linspace(0, T, num=int(T/dt))
plt.plot(Theta, Theta_d)
plt.axvline(gamma+alpha)
plt.axvline(gamma-alpha)
plt.show()
