import sympy as sp
import numpy as np
# from SumOfSquares import SOSProblem, poly_opt_prob
from cvxopt import matrix, solvers
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sympy import plot_implicit, symbols, Eq, And
from matplotlib.patches import Ellipse
import time

dt = 0.05
x_init_1 = [-0.75, -0.15]
x_init_2 = [0.4, 0.4]

def dynamics_1(state, u):
    x_1, x_2 = state[0], state[1]
    x_1new = x_1 + (x_1 + u[0][0])* dt 
    x_2new = x_2 + (-x_1 + 4 * x_2) * dt
    return [x_1new, x_2new]

def dynamics_2(state, u):
    x_1, x_2 = state[0], state[1]
    x_1new = x_1 + (x_2)* dt 
    x_2new = x_2 + (x_1 + u[1][0]) * dt
    return [x_1new, x_2new]    

def feedback_control(state):
    return -8 * state[0] + 30 * state[1]

def func_h_bar(state):
    k = 1.1575
    x_1 = state[0] - 0.1378
    x_2 = state[1] - 0
    P = matrix([[6.23, -26.7], [-26.7, 146.7]])
    return k - (P[0] * x_1 ** 2 + (P[1] + P[2]) * x_1 * x_2 + P[3] * x_2 ** 2)

def partial_h_bar(state):
    P = matrix([[6.23, -26.7], [-26.7, 146.7]])
    partial_x_1 = -2 * P[0] * state[0] - (P[1] + P[2]) * state[1]
    partial_x_2 = -2 * P[3] * state[1] - (P[1] + P[2]) * state[0]
    return matrix([[partial_x_1], [partial_x_2]])

def QP_CBF(u_norminal, state):
    alpha = 1000
    P = matrix([1.0])
    q = matrix([0.0])

    g_x = matrix([[1],[0]])
    
    g = partial_h_bar(state)*g_x.T
    hx = alpha * func_h_bar(state) + g[0]*u_norminal
    G = -g
    h = matrix([matrix(hx)]) 
    print(P,q,G,h)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P,q,G,h)
    u_cbf = np.array(sol['x'])
    return u_cbf

# def vaibility(state):


def main():
    T = 0.5
    t = 0
    use_cbf = True
    x = x_init_1

    X = []
    while t < T:
        X.append(x)
        u = feedback_control(x)
        if use_cbf == True:
            u += QP_CBF(u, x)
        x = dynamics_1(x,u)
        print(x)
        t += dt
    # print(np.transpose(X)[0])
    fig, ax = plt.subplots() 
    plt.plot(np.transpose(X)[0], np.transpose(X)[1])
    circle1 = plt.Circle((0,0), 1, color='blue', fill=False)
    ax.add_patch(circle1)
    ax = plt.gca()

    # ellipse = Ellipse(xy=(0, 0), width=1.86849, height=0.1747, 
                        # edgecolor='r', fc='None', lw=2)
    ellipse = Ellipse(xy=(0, 0), width=2.1, height=0.4, 
                        edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse)
    # t = np.linspace(-1,1, num=100)
    # ht = [func_h_bar([s,s]) for s in t]
    # plt.plot(t, t**2)

    plt.ylim(-1,1)
    plt.xlim(-1,2.5)
    plt.show()
    


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
