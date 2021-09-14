import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib
from cvxopt import matrix, solvers
import matplotlib.animation as animation
    
def UN_LN_controller(virtual_target, true_position, error):
    k1 = -40
    k2 = -100
    k3 = -2                                        # A Linear Controller for the Unicycle Model
    desire_speed = 5*np.pi/50
    desire_angular = 2*np.pi/50
    true_x_error = np.cos(true_position[2][0])*error[0][0] + np.sin(true_position[2][0])*error[1][0]
    true_y_error = np.cos(true_position[2][0])*error[1][0] - np.sin(true_position[2][0])*error[0][0]

    u_v = -k1 * true_x_error
    u_r = -k2 * np.sign(desire_speed) * true_y_error - k3 * error[2][0]

    u_true = np.array([[u_v], [u_r]])
    return u_true

def UN_controller(virtual_target, true_position, error,desire_speed = 5*np.pi/50,desire_angular = 2*np.pi/50):
    k2 = -10000                                       # A Nonlinear Controller for the Unicycle Model 
    true_x_error = np.cos(true_position[2][0])*error[0][0] + np.sin(true_position[2][0])*error[1][0]
    true_y_error = np.cos(true_position[2][0])*error[1][0] - np.sin(true_position[2][0])*error[0][0]

    u_v = -k_function(desire_speed, desire_angular) * true_x_error
    u_r = -k2 * np.sin(error[2][0]) * true_y_error / (error[2][0]+0.0001) - k_function(desire_speed, desire_angular) * error[2][0]

    u_true = np.array([[u_v], [u_r]])
    return u_true

def k_function(x1, x2):
    zeta, beta = -100, 100
    return 2*zeta*np.sqrt(x1**2+beta*x2**2)


def PI_controller(virtual_target, true_position, error, cum_error):
    kp1, ki1 = 15, 1
    kp2, ki2 = 10, 1
    true_x_error = error[0][0]
    true_y_error = error[1][0]
    true_theta = true_position[2][0]
    u_v = kp1 * (true_x_error * np.cos(true_theta) + true_y_error * np.sin(true_theta)) +\
        ki1 * (cum_error[0][0] * np.cos(true_theta) + cum_error[1][0] * np.sin(true_theta))
    u_r = kp2 * (error[2][0]) + ki2 * (cum_error[2][0])
    u_true = np.array([[u_v], [u_r]])
    return u_true

def g_id_leader(x):         # the trajectory of the leader
    if x <= 0.5:
        target = np.array([[2.5-2.5*np.cos(2*np.pi*x)], [2.5*np.sin(2*np.pi*x)],[np.pi/2-2*np.pi*x]])           ##desire x,y,theta theta is in radius
    else:
        target = np.array([[7.5-2.5*np.cos(-(x-0.5)*2*np.pi)], [2.5*np.sin(-(x-0.5)*2*np.pi)], [-np.pi/2+(x-0.5)*2*np.pi]])
    return target
    
def g_id_follower1(x):      #the trajectory of the follower
    if x <= 0.5:
        target = np.array([[5*np.sin(np.pi*x)], [5*np.cos(x * np.pi)], [-x*np.pi]])
    else:
        target = np.array([[8-3*np.cos(-(x-0.5)*2*np.pi)], [3*np.sin(-(x-0.5)*2*np.pi)], [-np.pi/2+(x-0.5)*2*np.pi]])
    return target

def g_id_follower2(x):      #the trajectory of the follower
    if x <= 0.5:
        target = np.array([[6.5-1.5*np.sin(np.pi*x)], [1.5*np.cos(x * np.pi)], [-np.pi+x*np.pi]])
    else:
        target = np.array([[8.5-3.5*np.cos(-(x-0.5)*2*np.pi)], [3.5*np.sin(-(x-0.5)*2*np.pi)], [-np.pi/2+(x-0.5)*2*np.pi]])
    return target

def dynamic(true_position, control_input, time_step, disturbance=True):
    mu = 0
    sigma = 0.1
    if disturbance == True:                                             #disturbance for dynamic model
        d = np.random.normal(mu, sigma)
    else:
        d = 0
    true_x = true_position[0][0] + control_input[0][0]*np.cos(true_position[2][0]) * 1/time_step + d
    true_y = true_position[1][0] + control_input[0][0]*np.sin(true_position[2][0]) * 1/time_step + d
    # true_theta = true_position[2][0] + control_input[0][0]*np.tan(control_input[1][0]) * 1/time_step
    true_theta = true_position[2][0] + control_input[1][0] * 1/time_step
    return np.array([[true_x], [true_y], [true_theta]])

def cbf(u_norminal, u_old, agent1_position, agent2_position, agent3_position, d_s=1):
    x, y, theta = agent1_position[0][0], agent1_position[1][0], agent1_position[2][0]
    g_x = matrix([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]], (3,2))
    partial_h_x1 = matrix([2*(x-agent2_position[0][0]), 2*(y-agent2_position[1][0]), 0], (1,3))
    partial_h_x2 = matrix([2*(x-agent3_position[0][0]), 2*(y-agent3_position[1][0]), 0], (1,3))
    h_x1 = (x-agent2_position[0][0])**2 + (y-agent2_position[1][0])**2 - d_s**2
    h_x2 = (x-agent3_position[0][0])**2 + (y-agent3_position[1][0])**2 - d_s**2
    alpha = 1000
    uv_max,uv_min,uomega_max,uomega_min = 20.0,-50,100,-100
    lipschitz_constrain = 5.0
    upper_lipschitz_cons = lipschitz_constrain + u_old[0][0] - u_norminal[0][0]
    lower_lipschitz_cons = u_norminal[0][0] - u_old[0][0] + lipschitz_constrain 
    # print(partial_h_x*g_x)
    g1 = partial_h_x1*g_x
    g2 = partial_h_x2*g_x
    # print(g.shape)
    P = matrix([[1.0,0.0],[0.0,1.0]])
    q = matrix([0.0,0.0])

    # G = -g
    # h = matrix([alpha * h_x]) + g * matrix(u_norminal)

    h_1 = alpha * h_x1 ** 3 + g1 * matrix(u_norminal)
    h_2 = alpha * h_x2 ** 3 + g2 * matrix(u_norminal)
    # G = matrix([[-g[0],1.0,-1.0,0.0,0.0], [0.0,0.0,0.0,1.0,-1.0]])
    # h = matrix([h_0[0][0],uv_max-u_norminal[0][0],-uv_min+u_norminal[0][0],uomega_max-u_norminal[1][0],-uomega_min+u_norminal[1][0]])

    # G = matrix([[-g[0],1.0,-1.0,0.0,0.0,1.0,-1.0], [0.0,0.0,0.0,1.0,-1.0,0.0,0.0]])
    # h = matrix([h_0[0][0],uv_max-u_norminal[0][0],-uv_min+u_norminal[0][0],uomega_max-u_norminal[1][0],-uomega_min+u_norminal[1][0],upper_lipschitz_cons,lower_lipschitz_cons])

    G = matrix([[-g1[0],-g2[0],1.0,-1.0,0.0,0.0,1.0,-1.0], [0.0,0.0,0.0,0.0,1.0,-1.0,0.0,0.0]])
    h = matrix([h_1[0][0],h_2[0][0],uv_max-u_norminal[0][0],-uv_min+u_norminal[0][0],uomega_max-u_norminal[1][0],-uomega_min+u_norminal[1][0],upper_lipschitz_cons,lower_lipschitz_cons])

    sol = solvers.qp(P,q,G,h)
    u_cbf = np.array(sol['x'])
    return u_cbf

def main():
    num_agent = 3
    lead_num_agent = 1
    follow_num_agent = num_agent - lead_num_agent

    x_init = np.array([[0.0],[0.0],[0.0]])
    kai_init = np.array([[1.0],[1.0]])
    k_p = 10
    k_i = 5
    d = np.array([[0.2],[-0.1]])
    P = np.array([[0.4,0.6],[0.4,0.6]])
    rho = 1
    time_step = 500
    T = np.linspace(0, 1, num=time_step+1)
    # L_matrix = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
    L_matrix = np.array([[1,-1,0],[-1,2,-1],[0,-1,1]])
    C = np.array([[0, 1, 0],[0, 0, 1]])

    x = x_init
    kai = kai_init
    X = np.zeros(shape=(time_step+1,3))
    U = np.zeros(shape=(time_step+1,3))
    X[0] = x.ravel()

    # Define variables
    Leader_virtual_target, Leader_true_x, Leader_control_input = np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,2))
    Follower1_virtual_target, Follower1_true_x, Follower1_control_input = np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,2))
    Follower2_virtual_target, Follower2_true_x, Follower2_control_input = np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,2))

    cum_error, error, follower1_error, follower2_error = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    leader_true_x, follower1_true_x, follower2_true_x = g_id_leader(x_init[0][0]), g_id_follower1(x_init[1][0]), g_id_follower2(x_init[2][0])
    
    # Initiallization
    Leader_virtual_target[0], Leader_true_x[0] = np.array([0,0,1.57]),  np.array([0,0,1.57])
    Follower1_virtual_target[0], Follower1_true_x[0] = follower1_true_x.T, follower1_true_x.T
    Follower2_virtual_target[0], Follower2_true_x[0] = follower2_true_x.T, follower2_true_x.T

    Use_cbf = True
    safe_distance = 0.8
    u_leader_old, u_follower1_old, u_follower2_old = np.array([[0],[0]]), np.array([[0],[0]]), np.array([[0],[0]])
    
    for t in range(time_step):
        u = - k_p * np.dot(L_matrix,x) + np.vstack((rho, kai))
        kai += -k_i * C.dot(L_matrix.dot(x))                        #kai the virtual time
        x = x + (u + np.vstack((0, d)))* 1 /time_step
        X[t+1] = x.ravel()
        U[t+1] = u.ravel()
        
        for i in range(num_agent):
            if i == 0:
                virtual_target =  g_id_leader(x[i][0])
                control_input = UN_controller(virtual_target, leader_true_x, error, desire_speed=10*np.pi/time_step)
                # print(control_input)
                if Use_cbf == True:
                    control_input += cbf(control_input, u_leader_old, leader_true_x, follower1_true_x, follower2_true_x, d_s=safe_distance)
                # print(control_input)
                leader_true_x = dynamic(leader_true_x, control_input, time_step, disturbance=False)
                error = virtual_target - leader_true_x
                cum_error += error
                u_leader_old = control_input

                Leader_virtual_target[t+1] = virtual_target.ravel()
                Leader_true_x[t+1] = leader_true_x.ravel()
                Leader_control_input[t+1] = control_input.ravel()

            if i == 1:
                virtual_target_1 = g_id_follower1(x[i][0])
                control_input_1 = UN_controller(virtual_target_1, follower1_true_x, follower1_error, desire_speed=8.5*np.pi/time_step)
                if Use_cbf == True:
                    control_input_1 += cbf(control_input_1, u_follower1_old, follower1_true_x, leader_true_x, follower2_true_x, d_s=safe_distance)
                follower1_true_x = dynamic(follower1_true_x, control_input_1, time_step, disturbance=0)
                follower1_error = virtual_target_1 - follower1_true_x
                u_follower1_old = control_input_1

                Follower1_virtual_target[t+1] = virtual_target_1.ravel()
                Follower1_true_x[t+1] = follower1_true_x.ravel()
                Follower1_control_input[t+1] = control_input_1.ravel()

            if i == 2:
                virtual_target_2 = g_id_follower2(x[i][0])
                control_input_2 = UN_controller(virtual_target_2, follower2_true_x, follower2_error, desire_speed=4.25*np.pi/time_step)
                if Use_cbf == True:
                    control_input_2 += cbf(control_input_2, u_follower2_old, follower2_true_x, leader_true_x, follower1_true_x, d_s=safe_distance)
                follower2_true_x = dynamic(follower2_true_x, control_input_2, time_step, disturbance=0)
                follower2_error = virtual_target_2 - follower2_true_x
                u_follower2_old = control_input_2

                Follower2_virtual_target[t+1] = virtual_target_2.ravel()
                Follower2_true_x[t+1] = follower2_true_x.ravel()
                Follower2_control_input[t+1] = control_input_2.ravel()

    fig1 = plt.figure()
    ax1 = plt.axes(xlim=(0, 15), ylim=(-5, 5))
    line1, = ax1.plot([], [], marker='o', color='r')
    traj1, = plt.plot([],[], color='r', alpha=0.5)
    line2, = ax1.plot([], [], marker='o', color='b')
    traj2, = plt.plot([],[], color='b', alpha=0.5)
    line3, = ax1.plot([], [], marker='o', color='g')
    traj3, = plt.plot([],[], color='g', alpha=0.5)


    plot_for_anim_leader = np.transpose(Leader_true_x)
    plot_for_anim_follower1 = np.transpose(Follower1_true_x)
    plot_for_anim_follower2 = np.transpose(Follower2_true_x)
    def animate(i):
        line1.set_data(plot_for_anim_leader[0][i], plot_for_anim_leader[1][i])
        traj1.set_data(plot_for_anim_leader[0][:i+1], plot_for_anim_leader[1][:i+1])
        line2.set_data(plot_for_anim_follower1[0][i], plot_for_anim_follower1[1][i])
        traj2.set_data(plot_for_anim_follower1[0][:i+1], plot_for_anim_follower1[1][:i+1])
        line3.set_data(plot_for_anim_follower2[0][i], plot_for_anim_follower2[1][i])
        traj3.set_data(plot_for_anim_follower2[0][:i+1], plot_for_anim_follower2[1][:i+1])
        return line1,traj1,line2,traj2,line3,traj3,
    rec1 = matplotlib.patches.Rectangle((1.5,-4),2,6,color='black')             # plot the building
    rec2 = matplotlib.patches.Rectangle((6.5,-1),2,6,color='black')
    ax1.add_patch(rec1)
    ax1.add_patch(rec2)
    anim = animation.FuncAnimation(fig1, animate, frames=501, interval=20, blit=True)
    plt.draw()
    plt.show()
    # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # print(cum_error)
    plt.plot(T, np.transpose(X)[0] - np.transpose(X)[1])
    plt.plot(T, np.transpose(X)[0] - np.transpose(X)[2])
    plt.plot(T, np.transpose(X)[2] - np.transpose(X)[1])
    plt.show()

    plt.plot(T, np.transpose(Leader_control_input)[0])
    plt.plot(T, np.transpose(Follower1_control_input)[0])
    plt.title('control_input')
    plt.show()

    Error = Leader_virtual_target-Leader_true_x
    plt.plot(T, np.transpose(Error)[0])
    plt.plot(T, np.transpose(Error)[1])
    plt.show()

    Distance1 = (np.transpose(Leader_true_x)[0]-np.transpose(Follower1_true_x)[0])**2 + (np.transpose(Leader_true_x)[1]-np.transpose(Follower1_true_x)[1])**2
    Distance2 = (np.transpose(Leader_true_x)[0]-np.transpose(Follower2_true_x)[0])**2 + (np.transpose(Leader_true_x)[1]-np.transpose(Follower2_true_x)[1])**2
    Distance3 = (np.transpose(Follower2_true_x)[0]-np.transpose(Follower1_true_x)[0])**2 + (np.transpose(Follower2_true_x)[1]-np.transpose(Follower1_true_x)[1])**2
    plt.plot(T, Distance1, label='distance1')
    plt.plot(T, Distance2, label='distance2')
    plt.plot(T, Distance3, label='distance3')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.axhline(y=safe_distance**2, color='r', linestyle='-')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.transpose(Leader_virtual_target)[0], np.transpose(Leader_virtual_target)[1])
    plt.plot(np.transpose(Leader_true_x)[0], np.transpose(Leader_true_x)[1])
    plt.plot(np.transpose(Follower1_virtual_target)[0], np.transpose(Follower1_virtual_target)[1])
    plt.plot(np.transpose(Follower1_true_x)[0], np.transpose(Follower1_true_x)[1])
    plt.plot(np.transpose(Follower2_virtual_target)[0], np.transpose(Follower2_virtual_target)[1])
    plt.plot(np.transpose(Follower2_true_x)[0], np.transpose(Follower2_true_x)[1])
    rec1 = matplotlib.patches.Rectangle((1.5,-4),2,6,color='black')             # plot the building
    rec2 = matplotlib.patches.Rectangle((6.5,-1),2,6,color='black')
    ax.add_patch(rec1)
    ax.add_patch(rec2)
    plt.show()
 

    
main()

