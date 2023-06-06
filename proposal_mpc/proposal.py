from casadi import *
import tqdm

N = 20 # number of control intervals



# ---- dynamic constraints --------
f = lambda x,u: vertcat(x[2], x[3], u[0], u[1]) # dx/dt = f(x,u)

dt = 0.05 # length of a control interval

circle_obstacles_1 = {'x': 0.5, 'y': 0.5, 'r': 0.5}
circle_obstacles_2 = {'x': -0.5, 'y': -0.5, 'r': 0.5}
circle_obstacles_3 = {'x': -1.0, 'y': 0.8, 'r': 0.5}

def distance_circle_obs(x, y, circle_obstacles):
    return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2

def solver_mpc(x_init, y_init, vx_init, vy_init):

    opti = Opti() # Optimization problem

    # ---- decision variables ---------
    X = opti.variable(4,N+1) # state trajectory
    pos_x = X[0,:]
    pos_y = X[1,:]
    vel_x = X[2,:]
    vel_y = X[3,:]

    U = opti.variable(2,N)   # control trajectory (throttle)
    acc_x = U[0,:]
    acc_y = U[1,:]


    # Objective term
    L = sumsqr(X) + sumsqr(U) # sum of QP terms

    # ---- objective          ---------
    opti.minimize(L) # race in minimal time 

    for k in range(N): # loop over control intervals
        # Runge-Kutta 4 integration
        k1 = f(X[:,k],         U[:,k])
        k2 = f(X[:,k]+dt/2*k1, U[:,k])
        k3 = f(X[:,k]+dt/2*k2, U[:,k])
        k4 = f(X[:,k]+dt*k3,   U[:,k])
        x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        opti.subject_to(X[:,k+1]==x_next) # close the gaps

    # ---- path constraints 1 -----------
    limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + 1.0
    limit_lower = lambda pos_x: sin(0.5*pi*pos_x) - 0.5
    opti.subject_to(limit_lower(pos_x)<=pos_y)
    opti.subject_to(pos_y<=limit_upper(pos_x))   # track speed limit

    # ---- path constraints 2 --------  
    # opti.subject_to(pos_y<=1.5)
    # opti.subject_to(pos_y>=-1.5)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_1) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_2) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_3) >= 0.0)

    # ---- input constraints --------
    opti.subject_to(opti.bounded(-10,U,10)) # control is limited

    # ---- boundary conditions --------
    opti.subject_to(pos_x[0]==x_init)
    opti.subject_to(pos_y[0]==y_init)   # start at position (0,0)
    opti.subject_to(vel_x[0]==vx_init)
    opti.subject_to(vel_y[0]==vy_init)   # start from stand-still 

    # ---- solve NLP              ------
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    
    opti.solver("ipopt", opts) # set numerical backend
    # opti.solver("ipopt") # set numerical backend
    sol = opti.solve()   # actual solve

    return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(vel_x[1]), sol.value(vel_y[1])

# ---- post-processing        ------
import matplotlib.pyplot as plt
x_0, y_0, vx_0, vy_0 = -3, 1, 1.0, -1.0
Epi = 500
x_log, y_log = [], [] 
for i in tqdm.tqdm(range(Epi)):
    try:
        x_0, y_0, vx_0, vy_0 = solver_mpc(x_0, y_0, vx_0, vy_0)
        x_log.append(x_0)
        y_log.append(y_0)
        if x_0 ** 2 + y_0 ** 2 < 0.01:
            break
    except RuntimeError:
        print('RuntimeError')
        break

print(x_0, y_0)

plt.plot(x_log, y_log, 'r-')
plt.plot(0,0,'bo')
plt.xlabel('pos_x')
plt.ylabel('pos_y')
plt.axis([-4.0, 4.0, -4.0, 4.0])

x = np.arange(-4,4,0.01)
y = np.sin(0.5 * pi * x) +1
plt.plot(x, y, 'g-', label='upper limit')
plt.plot(x, y-1.5, 'b-', label='lower limit')
plt.draw()
plt.pause(1)
input("<Hit Enter>")
plt.close()

target_circle1 = plt.Circle((circle_obstacles_1['x'], circle_obstacles_1['y']), circle_obstacles_1['r'], color='b', fill=False)
target_circle2 = plt.Circle((circle_obstacles_2['x'], circle_obstacles_2['y']), circle_obstacles_2['r'], color='b', fill=False)
target_circle3 = plt.Circle((circle_obstacles_3['x'], circle_obstacles_3['y']), circle_obstacles_3['r'], color='b', fill=False)
plt.gcf().gca().add_artist(target_circle1)
plt.gcf().gca().add_artist(target_circle2)
plt.gcf().gca().add_artist(target_circle3)
x = np.arange(-4,4,0.01)
y = 1.5 + 0*x
plt.plot(x, y, 'g-', label='upper limit')
plt.plot(x, y-3, 'b-', label='lower limit')
plt.draw()
plt.pause(1)
input("<Hit Enter>")
plt.close()

plt.show()

# def MPC_diff_init(x_0, y_0, vx_0, vy_0):
#     Epis = 1000
#     vx_init = vx_0
#     vy_init = vy_0
#     x_log, y_log = [], []
#     for t in tqdm.tqdm(range(Epis)):
#         try:
#             x_0, y_0, vx_0, vy_0 = solver_mpc(x_0, y_0, vx_0, vy_0)
#             x_log.append(x_0)
#             y_log.append(y_0)
#             if x_0 ** 2 + y_0 ** 2 < 0.01:
#                 return [1, vx_init, vy_init], x_log, y_log
#         except RuntimeError:
#             return [0, vx_init, vy_init], x_log, y_log
#     return [0, vx_init, vy_init], x_log, y_log

# VX = np.arange(-2.5, 2.5, 0.1)
# VY = np.arange(-2.5, 2.5, 0.1)
# LOG_vel = []
# LOG_traj = []
# ii = 0
# for vx in VX:
#     for vy in VY:
#         print("epsidoe", ii)
#         Data_vel, Data_tarj_x, Data_tarj_y = MPC_diff_init(-3, 1, vx, vy)
#         LOG_vel.append(Data_vel)
#         LOG_traj.append([Data_tarj_x, Data_tarj_y])
#         ii += 1

# print(LOG_vel)
# import pickle

# with open('LOG_vel_5.pkl', 'wb') as f:
#     pickle.dump(LOG_vel, f)

# with open('LOG_traj_5.pkl', 'wb') as f:
#     pickle.dump(LOG_traj, f)