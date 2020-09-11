import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib
from cvxopt import matrix, solvers
import matplotlib.animation as animation

def UN_controller(virtual_target, true_position, error,desire_speed = 5*np.pi/50,desire_angular = np.pi):
    k2 = -10000                                     # A Nonlinear Controller for the Unicycle Model 
    true_x_error = np.cos(true_position[2][0])*error[0][0] + np.sin(true_position[2][0])*error[1][0]
    true_y_error = np.cos(true_position[2][0])*error[1][0] - np.sin(true_position[2][0])*error[0][0]

    u_v = -k_function(desire_speed, desire_angular) * true_x_error
    u_r = -k2 * np.sin(error[2][0]) * true_y_error / (error[2][0]+0.0001) - k_function(desire_speed, desire_angular) * error[2][0]

    u_true = np.array([[u_v], [u_r]])
    return u_true

def k_function(x1, x2):
    zeta, beta = -1000, 100
    return 2*zeta*np.sqrt(x1**2+beta*x2**2)
    
def g_id_follower1(x):      #the trajectory of the follower 1
    if x < 1:
        target = np.array([[5-5.0*np.cos(np.pi*x)], [-6.0+5.0*np.sin(np.pi*x)],[-np.pi*x]])
    else:
        target = np.array([[5-5.0*np.cos(np.pi*1)], [-6.0+5.0*np.sin(np.pi*1)],[-np.pi*1]])
    return target

def line_trajectory(x,n,N):
    if x < 1:
        target = np.array([[5.0*np.cos(2*np.pi*n/N)-10*np.cos(2*np.pi*n/N)*x],[5.0*np.sin(2*np.pi*n/N)-10*np.sin(2*np.pi*n/N)*x],[2*np.pi*n/N-np.pi]])
    else:
        target = np.array([[-5.0*np.cos(2*np.pi*n/N)],[-5.0*np.sin(2*np.pi*n/N)],[2*np.pi*n/N-np.pi]])
    return target

# def line_trajectory(x,n,N):
#     if x < 1:
#         target = np.array([[5.0*(1-n/N)*np.cos(2*np.pi*x)],[5*(1-n/N)*np.sin(2*np.pi*x)],[2*np.pi*x]])
#     else:
#         target = np.array([[5.0*(1-n/N)*np.cos(2*np.pi)],[5*(1-n/N)*np.sin(2*np.pi)],[2*np.pi]])
#     return target

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

def cbf(u_norminal, u_old, Dict, index, t, num_agent, d_s):
    position_i = Dict["true{0}".format(index)][t]
    x, y, theta =  position_i[0], position_i[1], position_i[2]
    g_x = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    partial_h_x = np.zeros((num_agent-1,3))
    h_x = np.zeros((num_agent-1,1))
    for j in range(0, num_agent):
        if j < index:
            position_j = Dict["true{0}".format(j)][t-1]
            partial_h_x[j][0] = 2*(x-position_j[0])
            partial_h_x[j][1] = 2*(y-position_j[1])
            partial_h_x[j][2] = 0
            h_x[j] = (x-position_j[0])**2 + (y-position_j[1])**2 - d_s**2
        if j > index:
            position_j = Dict["true{0}".format(j)][t-1]
            print(position_j)
            partial_h_x[j-1][0] = 2*(x-position_j[0])
            partial_h_x[j-1][1] = 2*(y-position_j[1])
            partial_h_x[j-1][2] = 0
            h_x[j-1] = (x-position_j[0])**2 + (y-position_j[1])**2 - d_s**2

    alpha = 1000
    uv_max,uv_min,uomega_max,uomega_min = 50.0,-10.0,50.0,-50.0
    lipschitz_constrain = 10
    upper_lipschitz_cons = lipschitz_constrain + u_old[0][0] - u_norminal[0][0]
    lower_lipschitz_cons = u_norminal[0][0] - u_old[0][0] + lipschitz_constrain 

    g = partial_h_x.dot(g_x)
    g_prime = matrix([[1.0,-1.0,0.0,0.0,1.0,-1.0], [0.0,0.0,1.0,-1.0,0.0,0.0]])
    h_prime = matrix([uv_max-u_norminal[0][0],-uv_min+u_norminal[0][0],uomega_max-u_norminal[1][0],-uomega_min+u_norminal[1][0],upper_lipschitz_cons,lower_lipschitz_cons])

    P = matrix([[1.0,0.0],[0.0,1.0]])
    q = matrix([0.0,0.0])

    hx = alpha * h_x ** 3 + g.dot(u_norminal)

    # G = matrix([matrix(g),g_prime])
    # h = matrix([matrix(h),h_prime])
    G = matrix([matrix(-g)])
    h = matrix([matrix(hx)]) 

    sol = solvers.qp(P,q,G,h)
    u_cbf = np.array(sol['x'])
    return u_cbf

def main():
    num_agent = 4
    lead_num_agent = 1
    follow_num_agent = num_agent - lead_num_agent

    x_init = np.zeros((num_agent,1))
    kai_init = np.zeros((follow_num_agent,1))
    k_p = 10
    k_i = 5
    # d = np.random.randn(follow_num_agent,1)
    d = np.zeros((follow_num_agent,1))
    rho = 1
    time_step = time_step_1 = 2000
    time_step_size = 0.002
    
    L_matrix = np.full((num_agent, num_agent),-1) + (num_agent)*np.eye(num_agent)
    C = np.hstack((np.zeros((follow_num_agent,lead_num_agent)),np.eye(follow_num_agent)))

    x = x_init
    kai = kai_init
    X = np.zeros(shape=(time_step+1,num_agent))
    U = np.zeros(shape=(time_step+1,num_agent))
    X[0] = x.ravel()

    # Define variables
    Trajectory = {}
    for i in range(0,num_agent):
        initial_pos = line_trajectory(x_init[i][0], i, num_agent)
        trajectory = np.zeros(shape=(time_step+1,3))
        trajectory[0] = initial_pos.T
        real_traj = np.zeros(shape=(time_step+1,3))
        real_traj[0] = initial_pos.T
        Trajectory["target{0}".format(i)] = trajectory
        Trajectory["true{0}".format(i)] = real_traj
        Trajectory["control_input{0}".format(i)] = np.zeros(shape=(time_step+1,2))
        Trajectory["error{0}".format(i)] = np.zeros(shape=(time_step+1,3))
        Trajectory["old_control_input{0}".format(i)] = np.array([[0],[0]])
        Trajectory["initial{0}".format(i)] = initial_pos
        Trajectory["error_coord{0}".format(i)] = np.zeros(shape=(num_agent,1))

    Use_cbf = True
    Use_feedback = False
    # safe_distance = 10*np.pi/num_agent
    safe_distance = 0.5
    # safe_distance = 4/num_agent
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
            virtual_target = line_trajectory(x[i][0], i, num_agent)
            true_x = Trajectory["true{0}".format(i)][t].reshape((3,1))
            error = Trajectory["error{0}".format(i)][t].reshape((3,1))
            old_control_input = Trajectory["old_control_input{0}".format(i)]

            control_input = UN_controller(virtual_target, true_x, error, desire_speed=(1-i/num_agent)*0.07*np.pi, desire_angular=0.0002*np.pi)
            if Use_cbf == True:
                u_cbf = cbf(control_input, old_control_input, Trajectory, i, t, num_agent, d_s=safe_distance)
                control_input += u_cbf
            true_x = dynamic(true_x, control_input, time_step_size, disturbance=0)
            error = virtual_target - true_x
            
            

            Trajectory["target{0}".format(i)][t+1] = virtual_target.ravel()
            Trajectory["true{0}".format(i)][t+1] = true_x.ravel()            
            Trajectory["control_input{0}".format(i)][t+1] = control_input.ravel()
            Trajectory["error{0}".format(i)][t+1] = error.ravel()
            Trajectory["error_coord{0}".format(i)][i] = np.sqrt(error[0]**2+error[1]**2)
            Trajectory["old_control_input{0}".format(i)] = control_input

    T = np.linspace(0, 1, num=time_step+1)
    T_1 = np.linspace(0, 1, num=time_step_1+1)
    T_2 = np.linspace(0, time_step_1+1, num=time_step_1+1)
    # print(time_step_1, time_step)
    # fig1 = plt.figure()
    # ax1 = plt.axes(xlim=(0, 14), ylim=(-7, 7))
    # line1, = ax1.plot([], [], marker='o', color='r')
    # traj1, = plt.plot([],[], color='r', alpha=0.5)
    # line2, = ax1.plot([], [], marker='o', color='b')
    # traj2, = plt.plot([],[], color='b', alpha=0.5)
    # line3, = ax1.plot([], [], marker='o', color='g')
    # traj3, = plt.plot([],[], color='g', alpha=0.5)
    # line4, = ax1.plot([],[], marker='*', color='k')
    # traj4, = plt.plot([],[], color='k', alpha=0.5)
    # line5, = ax1.plot([],[], marker='*', color='k')
    # traj5, = plt.plot([],[], color='k', alpha=0.5)
    # line6, = ax1.plot([],[], marker='*', color='k')
    # traj6, = plt.plot([],[], color='k', alpha=0.5)


    # plot_for_anim_leader = np.transpose(Leader_true_x)
    # plot_for_anim_follower1 = np.transpose(Follower1_true_x)
    # plot_for_anim_follower2 = np.transpose(Follower2_true_x)
    # plot_for_anim_leader_target = np.transpose(Leader_virtual_target)
    # plot_for_anim_follower1_target = np.transpose(Follower1_virtual_target)
    # plot_for_anim_follower2_target = np.transpose(Follower2_virtual_target)
    # def animate(i):
    #     line1.set_data(plot_for_anim_leader[0][i], plot_for_anim_leader[1][i])
    #     traj1.set_data(plot_for_anim_leader[0][:i+1], plot_for_anim_leader[1][:i+1])
    #     line2.set_data(plot_for_anim_follower1[0][i], plot_for_anim_follower1[1][i])
    #     traj2.set_data(plot_for_anim_follower1[0][:i+1], plot_for_anim_follower1[1][:i+1])
    #     line3.set_data(plot_for_anim_follower2[0][i], plot_for_anim_follower2[1][i])
    #     traj3.set_data(plot_for_anim_follower2[0][:i+1], plot_for_anim_follower2[1][:i+1])
    #     line4.set_data(plot_for_anim_leader_target[0][i], plot_for_anim_leader_target[1][i])
    #     traj4.set_data(plot_for_anim_leader_target[0][:i+1], plot_for_anim_leader_target[1][:i+1])
    #     line5.set_data(plot_for_anim_follower1_target[0][i], plot_for_anim_follower1_target[1][i])
    #     traj5.set_data(plot_for_anim_follower1_target[0][:i+1], plot_for_anim_follower1_target[1][:i+1])
    #     line6.set_data(plot_for_anim_follower2_target[0][i], plot_for_anim_follower2_target[1][i])
    #     traj6.set_data(plot_for_anim_follower2_target[0][:i+1], plot_for_anim_follower2_target[1][:i+1])
    #     return line1,traj1,line2,traj2,line3,traj3,line4,traj4,line5,traj5,line6,traj6,
    # anim = animation.FuncAnimation(fig1, animate, frames=time_step_1+1, interval=20, blit=True)
    # plt.draw()
    # plt.show()
    # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    # plt.plot(T, np.transpose(X)[0] - np.transpose(X)[1])
    # plt.plot(T, np.transpose(X)[0] - np.transpose(X)[2])
    # plt.plot(T, np.transpose(X)[2] - np.transpose(X)[1])
    # plt.show()

    # plt.plot(T, np.transpose(X)[0])
    # plt.plot(T, np.transpose(X)[1])
    # plt.plot(T, np.transpose(X)[2])
    # plt.show()

    ax1 = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
    for i in range(num_agent):
        xx = np.transpose(Trajectory["true{0}".format(i)])[0]
        yy = np.transpose(Trajectory["true{0}".format(i)])[1]
        xxd = np.transpose(Trajectory["target{0}".format(i)])[0]
        yyd = np.transpose(Trajectory["target{0}".format(i)])[1]
        plt.plot(xx, yy, color='blue')
        plt.plot(xxd, yyd, color='green')
        # print(xx,yy)
    # plt.plot(np.transpose(Trajectory["true0"])[0], np.transpose(Trajectory["true0"])[1])
    # plt.plot(np.transpose(Trajectory["target0"])[0], np.transpose(Trajectory["target0"])[1])
    plt.show()

    print(Trajectory["true1"])

    # plt.plot(T_2, np.transpose(Leader_control_input)[0])
    # plt.plot(T_2, np.transpose(Follower1_control_input)[0])
    # plt.plot(T_2, np.transpose(Follower2_control_input)[0])
    # plt.title('control_input')
    # plt.show()

    # Error = Leader_virtual_target-Leader_true_x
    # plt.plot(T_1, np.transpose(Error)[0])
    # plt.plot(T_1, np.transpose(Error)[1])
    # plt.show()

    # Distance1 = (np.transpose(Leader_true_x)[0]-np.transpose(Follower1_true_x)[0])**2 + (np.transpose(Leader_true_x)[1]-np.transpose(Follower1_true_x)[1])**2
    # Distance2 = (np.transpose(Leader_true_x)[0]-np.transpose(Follower2_true_x)[0])**2 + (np.transpose(Leader_true_x)[1]-np.transpose(Follower2_true_x)[1])**2
    # Distance3 = (np.transpose(Follower2_true_x)[0]-np.transpose(Follower1_true_x)[0])**2 + (np.transpose(Follower2_true_x)[1]-np.transpose(Follower1_true_x)[1])**2
    # plt.plot(T_1, Distance1, label='distance1')
    # plt.plot(T_1, Distance2, label='distance2')
    # # plt.plot(T_1, Distance3, label='distance3')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # plt.axhline(y=safe_distance**2, color='r', linestyle='-')
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.transpose(Leader_virtual_target)[0], np.transpose(Leader_virtual_target)[1])
    # plt.plot(np.transpose(Leader_true_x)[0], np.transpose(Leader_true_x)[1])
    # plt.plot(np.transpose(Follower1_virtual_target)[0], np.transpose(Follower1_virtual_target)[1])
    # plt.plot(np.transpose(Follower1_true_x)[0], np.transpose(Follower1_true_x)[1])
    # plt.plot(np.transpose(Follower2_virtual_target)[0], np.transpose(Follower2_virtual_target)[1])
    # plt.plot(np.transpose(Follower2_true_x)[0], np.transpose(Follower2_true_x)[1])
    # plt.show()
    
main()

