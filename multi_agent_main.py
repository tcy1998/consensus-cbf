import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib
from cvxopt import matrix, solvers
import matplotlib.animation as animation

def UN_controller(virtual_target, true_position, error,desire_speed = 5*np.pi/50,desire_angular = 2*np.pi/50):
    k2 = -10000                                      # A Nonlinear Controller for the Unicycle Model 
    true_x_error = np.cos(true_position[2][0])*error[0][0] + np.sin(true_position[2][0])*error[1][0]
    true_y_error = np.cos(true_position[2][0])*error[1][0] - np.sin(true_position[2][0])*error[0][0]

    u_v = -k_function(desire_speed, desire_angular) * true_x_error
    u_r = -k2 * np.sin(error[2][0]) * true_y_error / (error[2][0]+0.0001) - k_function(desire_speed, desire_angular) * error[2][0]

    u_true = np.array([[u_v], [u_r]])
    return u_true

def k_function(x1, x2):
    zeta, beta = -100, 100
    return 2*zeta*np.sqrt(x1**2+beta*x2**2)

def g_id_leader(x):         # the trajectory of the leader
    if x < 1:
        target = np.array([[10.0*x], [0.0],[0.0]])           ##desire x,y,theta theta is in radius
    else:
        target = np.array([[10.0*1], [0.0],[0.0]])
    return target
    
def g_id_follower1(x):      #the trajectory of the follower 1
    if x < 1:
        target = np.array([[5-5.0*np.cos(np.pi*x)], [-6.0+5.0*np.sin(np.pi*x)],[-np.pi*x]])
    else:
        target = np.array([[5-5.0*np.cos(np.pi*1)], [-6.0+5.0*np.sin(np.pi*1)],[-np.pi*1]])
    return target

def g_id_follower2(x):      #the trajectory of the follower 2
    if x < 1: 
        target = np.array([[5-5.0*np.cos(np.pi*x)], [6.0-5.0*np.sin(np.pi*x)],[np.pi*x]])
    else:
        target = np.array([[5-5.0*np.cos(np.pi*1)], [6.0-5.0*np.sin(np.pi*1)],[np.pi*1]])
    return target

def dynamic(true_position, control_input, time_step_size, disturbance=True):
    mu = 0
    sigma = 0.01
    if disturbance == True:                                             #disturbance for dynamic model
        d = np.random.normal(mu, sigma)
    else:
        d = 0
    true_x = true_position[0][0] + control_input[0][0]*np.cos(true_position[2][0]) * time_step_size + d
    true_y = true_position[1][0] + control_input[0][0]*np.sin(true_position[2][0]) * time_step_size + d
    # true_theta = true_position[2][0] + control_input[0][0]*np.tan(control_input[1][0]) * 1/time_step
    true_theta = true_position[2][0] + control_input[1][0] * time_step_size
    return np.array([[true_x], [true_y], [true_theta]])

def cbf(u_norminal, u_old, agent1_position, agent2_position, agent3_position, d_s=1):
    x, y, theta = agent1_position[0][0], agent1_position[1][0], agent1_position[2][0]
    g_x = matrix([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]], (3,2))
    partial_h_x1 = matrix([2*(x-agent2_position[0][0]), 2*(y-agent2_position[1][0]), 0], (1,3))
    partial_h_x2 = matrix([2*(x-agent3_position[0][0]), 2*(y-agent3_position[1][0]), 0], (1,3))
    h_x1 = (x-agent2_position[0][0])**2 + (y-agent2_position[1][0])**2 - d_s**2
    h_x2 = (x-agent3_position[0][0])**2 + (y-agent3_position[1][0])**2 - d_s**2
    alpha = 1000
    uv_max,uv_min,uomega_max,uomega_min = 50.0,-10.0,50.0,-50.0
    lipschitz_constrain = 10
    upper_lipschitz_cons = lipschitz_constrain + u_old[0][0] - u_norminal[0][0]
    lower_lipschitz_cons = u_norminal[0][0] - u_old[0][0] + lipschitz_constrain 

    g1 = partial_h_x1*g_x
    g2 = partial_h_x2*g_x

    P = matrix([[1.0,0.0],[0.0,1.0]])
    q = matrix([0.0,0.0])

    h_1 = alpha * h_x1 ** 3 + g1 * matrix(u_norminal)
    h_2 = alpha * h_x2 ** 3 + g2 * matrix(u_norminal)

    G = matrix([[-g1[0],-g2[0],1.0,-1.0,0.0,0.0,1.0,-1.0], [0.0,0.0,0.0,0.0,1.0,-1.0,0.0,0.0]])
    h = matrix([h_1[0][0],h_2[0][0],uv_max-u_norminal[0][0],-uv_min+u_norminal[0][0],uomega_max-u_norminal[1][0],-uomega_min+u_norminal[1][0],upper_lipschitz_cons,lower_lipschitz_cons])

    sol = solvers.qp(P,q,G,h)
    u_cbf = np.array(sol['x'])
    return u_cbf

def main():
    num_agent = 3
    lead_num_agent = 1
    follow_num_agent = num_agent - lead_num_agent

    x_init = np.zeros((num_agent,1))
    kai_init = np.zeros((follow_num_agent,1))
    k_p = 10
    k_i = 5
    # d = np.array([[2],[-0.1]])
    d = np.random.randn(follow_num_agent,1)
    rho = 1
    time_step = time_step_1 = 600
    time_step_size = 0.002
    
    L_matrix = np.full((num_agent, num_agent),-1) + (num_agent)*np.eye(num_agent)
    C = np.hstack((np.zeros((follow_num_agent,lead_num_agent)),np.eye(follow_num_agent)))

    x = x_init
    kai = kai_init
    X = np.zeros(shape=(time_step+1,num_agent))
    U = np.zeros(shape=(time_step+1,num_agent))
    X[0] = x.ravel()

    # Define variables
    Leader_virtual_target, Leader_true_x, Leader_control_input = np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,2))
    Follower1_virtual_target, Follower1_true_x, Follower1_control_input = np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,2))
    Follower2_virtual_target, Follower2_true_x, Follower2_control_input = np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,3)), np.zeros(shape=(time_step+1,2))

    error, follower1_error, follower2_error = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    leader_true_x, follower1_true_x, follower2_true_x = g_id_leader(x_init[0][0]), g_id_follower1(x_init[1][0]), g_id_follower2(x_init[2][0])
    
    # Initiallization
    Leader_virtual_target[0], Leader_true_x[0] = leader_true_x.T,  leader_true_x.T
    Follower1_virtual_target[0], Follower1_true_x[0] = follower1_true_x.T, follower1_true_x.T
    Follower2_virtual_target[0], Follower2_true_x[0] = follower2_true_x.T, follower2_true_x.T

    Use_cbf = True
    use_to_the_end = False
    Use_feedback = True
    safe_distance = 3.0
    u_leader_old, u_follower1_old, u_follower2_old = np.array([[0],[0]]), np.array([[0],[0]]), np.array([[0],[0]])
    Error_all = np.array([[0],[0],[0]])
    k_e = 10
    
    for t in range(time_step):
        u = - k_p * np.dot(L_matrix,x) + np.vstack((rho, kai))
        if Use_feedback == True:
            u_change = -k_e * Error_all
            u += u_change
        kai += -k_i * C.dot(L_matrix.dot(x))                        #kai the virtual time
        x = x + (u + np.vstack((np.zeros((lead_num_agent,1)), d)))* time_step_size
        X[t+1] = x.ravel()
        U[t+1] = u.ravel()
        
        for i in range(num_agent):
            if i == 0:
                virtual_target =  g_id_leader(x[i][0])
                control_input = UN_controller(virtual_target, leader_true_x, error, desire_speed=15/time_step)
                # print(control_input)
                if Use_cbf == True:
                    control_input += cbf(control_input, u_leader_old, leader_true_x, follower1_true_x, follower2_true_x, d_s=safe_distance)
                # print(control_input)
                leader_true_x = dynamic(leader_true_x, control_input, time_step_size, disturbance=0)
                error = virtual_target - leader_true_x
                cum_error += error
                u_leader_old = control_input

                Leader_virtual_target[t+1] = virtual_target.ravel()
                Leader_true_x[t+1] = leader_true_x.ravel()
                Leader_control_input[t+1] = control_input.ravel()
                # Error_all[0] = np.sqrt(error[0]**2+error[1]**2)
                Error_all[0] = np.maximum(abs(error[0]), abs(error[1]))

            if i == 1:
                virtual_target_1 = g_id_follower1(x[i][0])
                control_input_1 = UN_controller(virtual_target_1, follower1_true_x, follower1_error, desire_speed=10*np.pi/time_step)
                if Use_cbf == True:
                    control_input_1 += cbf(control_input_1, u_follower1_old, follower1_true_x, leader_true_x, follower2_true_x, d_s=safe_distance)
                follower1_true_x = dynamic(follower1_true_x, control_input_1, time_step_size, disturbance=0)
                follower1_error = virtual_target_1 - follower1_true_x
                u_follower1_old = control_input_1

                Follower1_virtual_target[t+1] = virtual_target_1.ravel()
                Follower1_true_x[t+1] = follower1_true_x.ravel()
                Follower1_control_input[t+1] = control_input_1.ravel()
                Error_all[1] = np.sqrt(follower1_error[0]**2+follower1_error[1]**2)

            if i == 2:
                virtual_target_2 = g_id_follower2(x[i][0])
                control_input_2 = UN_controller(virtual_target_2, follower2_true_x, follower2_error, desire_speed=10*np.pi/time_step)
                if Use_cbf == True:
                    control_input_2 += cbf(control_input_2, u_follower2_old, follower2_true_x, leader_true_x, follower1_true_x, d_s=safe_distance)
                follower2_true_x = dynamic(follower2_true_x, control_input_2, time_step_size, disturbance=0)
                follower2_error = virtual_target_2 - follower2_true_x
                u_follower2_old = control_input_2

                Follower2_virtual_target[t+1] = virtual_target_2.ravel()
                Follower2_true_x[t+1] = follower2_true_x.ravel()
                Follower2_control_input[t+1] = control_input_2.ravel()
                Error_all[2] = np.sqrt(follower2_error[0]**2+follower2_error[1]**2)
        
    if use_to_the_end == True:
        while abs(error[1][0]) > 1e-2 or abs(follower1_error[1][0]) > 1e-2 or abs(follower2_error[1][0]) > 1e-2 or abs(error[0][0]) > 1e-2 or abs(follower1_error[0][0]) > 1e-2 or abs(follower2_error[0][0]) > 1e-2:
            control_input = UN_controller(virtual_target, leader_true_x, error, desire_speed=10*np.pi/time_step)
            control_input_1 = UN_controller(virtual_target_1, follower1_true_x, follower1_error, desire_speed=8.5*np.pi/time_step)
            control_input_2 = UN_controller(virtual_target_2, follower2_true_x, follower2_error, desire_speed=4.25*np.pi/time_step)
            if Use_cbf == True:
                control_input += cbf(control_input, u_leader_old, leader_true_x, follower1_true_x, follower2_true_x, d_s=safe_distance)
                control_input_1 += cbf(control_input_1, u_follower1_old, follower1_true_x, leader_true_x, follower2_true_x, d_s=safe_distance)
                control_input_2 += cbf(control_input_2, u_follower2_old, follower2_true_x, leader_true_x, follower1_true_x, d_s=safe_distance)
            leader_true_x = dynamic(leader_true_x, control_input, time_step, disturbance=0)
            follower1_true_x = dynamic(follower1_true_x, control_input_1, time_step, disturbance=0)
            follower2_true_x = dynamic(follower2_true_x, control_input_2, time_step, disturbance=0)
            error = virtual_target - leader_true_x
            follower1_error = virtual_target_1 - follower1_true_x
            follower2_error = virtual_target_2 - follower2_true_x
            u_leader_old = control_input
            u_follower1_old = control_input_1
            u_follower2_old = control_input_2

            time_step_1 += 1
            Leader_true_x = np.concatenate((Leader_true_x, np.transpose(leader_true_x)),axis=0)
            Leader_control_input = np.concatenate((Leader_control_input, np.transpose(control_input)),axis=0)
            Leader_virtual_target = np.concatenate((Leader_virtual_target, np.transpose(virtual_target)),axis=0)
            Follower1_true_x = np.concatenate((Follower1_true_x, np.transpose(follower1_true_x)),axis=0)
            Follower1_control_input = np.concatenate((Follower1_control_input, np.transpose(control_input_1)),axis=0)
            Follower2_true_x = np.concatenate((Follower2_true_x, np.transpose(follower2_true_x)),axis=0)
            Follower2_control_input = np.concatenate((Follower2_control_input, np.transpose(control_input_2)),axis=0)

    T = np.linspace(0, 1, num=time_step+1)
    T_1 = np.linspace(0, 1, num=time_step_1+1)
    T_2 = np.linspace(0, time_step_1+1, num=time_step_1+1)
    print(time_step_1, time_step)
    fig1 = plt.figure()
    ax1 = plt.axes(xlim=(0, 14), ylim=(-7, 7))
    line1, = ax1.plot([], [], marker='o', color='r')
    traj1, = plt.plot([],[], color='r', alpha=0.5)
    line2, = ax1.plot([], [], marker='o', color='b')
    traj2, = plt.plot([],[], color='b', alpha=0.5)
    line3, = ax1.plot([], [], marker='o', color='g')
    traj3, = plt.plot([],[], color='g', alpha=0.5)
    line4, = ax1.plot([],[], marker='*', color='k')
    traj4, = plt.plot([],[], color='k', alpha=0.5)
    line5, = ax1.plot([],[], marker='*', color='k')
    traj5, = plt.plot([],[], color='k', alpha=0.5)
    line6, = ax1.plot([],[], marker='*', color='k')
    traj6, = plt.plot([],[], color='k', alpha=0.5)


    plot_for_anim_leader = np.transpose(Leader_true_x)
    plot_for_anim_follower1 = np.transpose(Follower1_true_x)
    plot_for_anim_follower2 = np.transpose(Follower2_true_x)
    plot_for_anim_leader_target = np.transpose(Leader_virtual_target)
    plot_for_anim_follower1_target = np.transpose(Follower1_virtual_target)
    plot_for_anim_follower2_target = np.transpose(Follower2_virtual_target)
    def animate(i):
        line1.set_data(plot_for_anim_leader[0][i], plot_for_anim_leader[1][i])
        traj1.set_data(plot_for_anim_leader[0][:i+1], plot_for_anim_leader[1][:i+1])
        line2.set_data(plot_for_anim_follower1[0][i], plot_for_anim_follower1[1][i])
        traj2.set_data(plot_for_anim_follower1[0][:i+1], plot_for_anim_follower1[1][:i+1])
        line3.set_data(plot_for_anim_follower2[0][i], plot_for_anim_follower2[1][i])
        traj3.set_data(plot_for_anim_follower2[0][:i+1], plot_for_anim_follower2[1][:i+1])
        line4.set_data(plot_for_anim_leader_target[0][i], plot_for_anim_leader_target[1][i])
        traj4.set_data(plot_for_anim_leader_target[0][:i+1], plot_for_anim_leader_target[1][:i+1])
        line5.set_data(plot_for_anim_follower1_target[0][i], plot_for_anim_follower1_target[1][i])
        traj5.set_data(plot_for_anim_follower1_target[0][:i+1], plot_for_anim_follower1_target[1][:i+1])
        line6.set_data(plot_for_anim_follower2_target[0][i], plot_for_anim_follower2_target[1][i])
        traj6.set_data(plot_for_anim_follower2_target[0][:i+1], plot_for_anim_follower2_target[1][:i+1])
        return line1,traj1,line2,traj2,line3,traj3,line4,traj4,line5,traj5,line6,traj6,
    anim = animation.FuncAnimation(fig1, animate, frames=time_step_1+1, interval=20, blit=True)
    plt.draw()
    plt.show()
    # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.plot(T, np.transpose(X)[0] - np.transpose(X)[1])
    plt.plot(T, np.transpose(X)[0] - np.transpose(X)[2])
    plt.plot(T, np.transpose(X)[2] - np.transpose(X)[1])
    plt.show()

    plt.plot(T, np.transpose(X)[0])
    plt.plot(T, np.transpose(X)[1])
    plt.plot(T, np.transpose(X)[2])
    plt.show()

    plt.plot(T_2, np.transpose(Leader_control_input)[0])
    plt.plot(T_2, np.transpose(Follower1_control_input)[0])
    plt.plot(T_2, np.transpose(Follower2_control_input)[0])
    plt.title('control_input')
    plt.show()

    Error = Leader_virtual_target-Leader_true_x
    plt.plot(T_1, np.transpose(Error)[0])
    plt.plot(T_1, np.transpose(Error)[1])
    plt.show()

    Distance1 = (np.transpose(Leader_true_x)[0]-np.transpose(Follower1_true_x)[0])**2 + (np.transpose(Leader_true_x)[1]-np.transpose(Follower1_true_x)[1])**2
    Distance2 = (np.transpose(Leader_true_x)[0]-np.transpose(Follower2_true_x)[0])**2 + (np.transpose(Leader_true_x)[1]-np.transpose(Follower2_true_x)[1])**2
    Distance3 = (np.transpose(Follower2_true_x)[0]-np.transpose(Follower1_true_x)[0])**2 + (np.transpose(Follower2_true_x)[1]-np.transpose(Follower1_true_x)[1])**2
    plt.plot(T_1, Distance1, label='distance1')
    plt.plot(T_1, Distance2, label='distance2')
    # plt.plot(T_1, Distance3, label='distance3')
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
    plt.show()
    
main()

