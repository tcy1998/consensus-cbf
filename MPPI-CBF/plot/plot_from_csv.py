from cvxpy.atoms.norm import norm
from numpy import loadtxt
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape
from mppi_trajectory_follow import MPPI
from cvxopt import matrix
import matplotlib.cm as cm
from matplotlib.colors import Normalize
# import matplotlib as mpl

def calculate_cost(data):
    cost = 0
    # print(np.shape(data)[0])
    for i in range(np.shape(data)[0]):
        cost += (data[i][0] - 4.0) ** 2 + (data[i][1] - 4.0)**2
        if (data[i][0] - obstcle_x) ** 2 + (data[i][1] - obstcle_y)**2 < r **2:
            cost += 10   
    return cost

def calculate_multi_cost(data):
    cost = 0
    # print(np.shape(data)[0])
    for i in range(np.shape(data)[0]):
        cost += (data[i][0] - 4.0) ** 2 + (data[i][1] - 4.0)**2
        for j in range(3):
            if (data[i][0] - Obstacle_X[j]) ** 2 + (data[i][1] - Obstacle_Y[j])**2 < R[j] **2:
                cost += 10   
    return cost

def state_cost(data):
    costs = []
    for i in range(np.shape(data)[0]):
        cost = (data[i][0] - 4.0) ** 2 + (data[i][1] - 4.0)**2
        costs.append(cost)
    if (data[i][0] - obstcle_x) ** 2 + (data[i][1] - obstcle_y)**2 < r **2:
        cost += 10000
    return costs

def plot_time(data):
    length = np.shape(data)[0]
    t_linspace = np.linspace(0, tau*length, num=length)
    return t_linspace

obstcle_x = 2.2
obstcle_y = 2.0
r =0.5
target = [4.0,4.0]

Obstacle_X = matrix([1.5, 1.5, 3.5])
Obstacle_Y = matrix([1.0, 3.3, 2.5])
R = matrix([1.0, 1.0, 1.0])

tau = 0.05

#  Orignal MPPI sample plot based on the costs 
def sample_data_plot(dataname):
    sample_data_CBF = loadtxt(dataname, delimiter=',')
    print(np.shape(sample_data_CBF))
    sample_data_CBF_reshape = sample_data_CBF.reshape(sample_data_CBF.shape[0], sample_data_CBF.shape[1] // 3,3)
    sample_data_CBF_cost = []
    for i in range(len(sample_data_CBF_reshape)):
        cost1 = calculate_cost(sample_data_CBF_reshape[i])
        sample_data_CBF_cost.append(cost1)

    # ax = plt.subplots()
    plt.figure(figsize=(10,8))
    norm = Normalize(min(sample_data_CBF_cost),max(sample_data_CBF_cost))
    print(np.shape(sample_data_CBF_cost), norm)
    for i in range(len(sample_data_CBF_reshape)):
        plt.plot(np.transpose(sample_data_CBF_reshape[i])[0], np.transpose(sample_data_CBF_reshape[i])[1], color=cm.coolwarm(norm(sample_data_CBF_cost[i])))

    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    plt.colorbar(sm)
    circle1 = plt.Circle((obstcle_x, obstcle_y), r, color='k', fill=False)
    ax = plt.gca()
    ax.add_artist(circle1)
    ax.grid(True)
    ax.set_aspect(1) 

    plt.xlim([1,4])
    plt.ylim([1,4])
    plt.show()
    # randomly selcet 20 samples from 100 samples Orignal MPPI
    # random_indices = np.random.choice(np.shape(sample_data_CBF_reshape)[0], size=20, replace=False)
    # print(random_indices,np.shape(sample_data_CBF_reshape))

    # random_samples = sample_data_CBF_reshape[random_indices,:]
    # random_samples_cost = []

    # # print(np.shape(random_samples))
    # for i in range(len(random_samples)):
    #     cost2 = calculate_cost(random_samples[i])
    #     random_samples_cost.append(cost2)
    # random_samples_norm = Normalize(min(random_samples_cost), max(random_samples_cost))
    # plt.figure(figsize=(8,8))
    # for i in range(len(random_samples)):
    #     plt.plot(np.transpose(random_samples[i])[0], np.transpose(random_samples[i])[1], color=cm.coolwarm(random_samples_norm(random_samples_cost[i])))
    # circle1 = plt.Circle((obstcle_x, obstcle_y), r, color='k', fill=False)
    # ax = plt.gca()
    # ax.add_artist(circle1)

    # plt.xlim([0,4])
    # plt.ylim([0,4])
    # plt.show()

def sample_multi_data_plot(dataname):
    sample_data_CBF = loadtxt(dataname, delimiter=',')
    print(np.shape(sample_data_CBF))
    sample_data_CBF_reshape = sample_data_CBF.reshape(sample_data_CBF.shape[0], sample_data_CBF.shape[1] // 3,3)
    sample_data_CBF_cost = []
    for i in range(len(sample_data_CBF_reshape)):
        cost1 = calculate_multi_cost(sample_data_CBF_reshape[i])
        sample_data_CBF_cost.append(cost1)

    # ax = plt.subplots()
    plt.figure(figsize=(10,8))
    norm = Normalize(min(sample_data_CBF_cost),max(sample_data_CBF_cost))
    print(np.shape(sample_data_CBF_cost), norm)
    for i in range(len(sample_data_CBF_reshape)):
        plt.plot(np.transpose(sample_data_CBF_reshape[i])[0], np.transpose(sample_data_CBF_reshape[i])[1], color=cm.coolwarm(norm(sample_data_CBF_cost[i])))

    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    plt.colorbar(sm)

    for i in range(len(R)):
        circle1 = plt.Circle((Obstacle_X[i], Obstacle_Y[i]), R[i], color='k', fill=False)
        ax = plt.gca()
        ax.add_artist(circle1)
    plt.xlim([0,5.5])
    plt.ylim([-1,4.5])
    ax.grid(True)
    ax.set_aspect(1)
    plt.show()

sample_data_plot('100sample_20steps_single_MPPI_7timestep.csv')
sample_data_plot('100sample_20steps_single_CBF_7timestep.csv')



sample_multi_data_plot('100sample_40steps_multi_MPPI_7timestep.csv')
sample_multi_data_plot('100sample_20steps_multi_CBF_7timestep_1.csv')






CBF_500 = loadtxt('500sample_single_CBF.csv',delimiter=',')
CBF_200 = loadtxt('200sample_single_CBF.csv', delimiter=',')
CBF_100 = loadtxt('100sample_single_CBF.csv', delimiter=',')
CBF_50 = loadtxt('50sample_single_CBF.csv', delimiter=',')


plt.figure(figsize=(8,8))
plt.plot(np.transpose(CBF_500)[0], np.transpose(CBF_500)[1],label='500 samples')
plt.plot(np.transpose(CBF_200)[0], np.transpose(CBF_200)[1],label='200 samples')
plt.plot(np.transpose(CBF_100)[0], np.transpose(CBF_100)[1],label='100 samples')
plt.plot(np.transpose(CBF_50)[0], np.transpose(CBF_50)[1],label='50 samples')
circle1 = plt.Circle((obstcle_x, obstcle_y), r, color='k', fill=False)
ax = plt.gca()
ax.add_artist(circle1)
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
target_circle = plt.Circle((target[0], target[1]), 0.2, color='b', fill=False)
target_circle_1 = plt.Circle((target[0], target[1]), 0.1, color='b', fill=False)
ax.add_artist(target_circle)
ax.add_artist(target_circle_1)
plt.plot([target[0]], [target[1]], '+', color='b', markersize=50)
plt.xlim([0,4.5])
plt.ylim([0,4.5])
ax.grid(True)
ax.set_aspect(1)
plt.show()

# cost_CBF_500 = calculate_cost(CBF_500)

multi_CBF_100_20 = loadtxt('100sample_multi_CBF.csv', delimiter=',')
multi_CBF_50_20 = loadtxt('50sample_20steps_multi_CBF.csv', delimiter=',')
multi_MPPI_50_40 = loadtxt('50sample_40steps_multi_MPPI.csv', delimiter=',')
multi_MPPI_100_40 = loadtxt('100sample_40steps_multi_MPPI.csv',delimiter=',')

plt.figure(figsize=(8,8))
plt.plot(np.transpose(multi_CBF_100_20)[0], np.transpose(multi_CBF_100_20)[1],label='100 samples MPPI-CBF')
plt.plot(np.transpose(multi_CBF_50_20)[0], np.transpose(multi_CBF_50_20)[1],label='50 samples MPPI-CBF')
plt.plot(np.transpose(multi_MPPI_100_40)[0], np.transpose(multi_MPPI_100_40)[1],label='100 samples MPPI')
plt.plot(np.transpose(multi_MPPI_50_40)[0], np.transpose(multi_MPPI_50_40)[1],label='50 samples MPPI')


for i in range(len(R)):
    circle1 = plt.Circle((Obstacle_X[i], Obstacle_Y[i]), R[i], color='k', fill=False)
    ax = plt.gca()
    ax.add_artist(circle1)
plt.xlim([0,5.5])
plt.ylim([-1,4.5])
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
target_circle = plt.Circle((target[0], target[1]), 0.2, color='b', fill=False)
target_circle_1 = plt.Circle((target[0], target[1]), 0.1, color='b', fill=False)
ax.add_artist(target_circle)
ax.add_artist(target_circle_1)
plt.plot([target[0]], [target[1]], '+', color='b', markersize=50)
ax.grid(True)
ax.set_aspect(1)
plt.show()

MPPI_500 = loadtxt('500sample_20steps_single_MPPI.csv',delimiter=',')
MPPI_200 = loadtxt('200sample_20steps_single_MPPI.csv', delimiter=',')
MPPI_100 = loadtxt('100sample_20steps_single_MPPI.csv', delimiter=',')
MPPI_50 = loadtxt('50sample_20steps_single_MPPI.csv', delimiter=',')

plt.figure(figsize=(8,8))
plt.plot(np.transpose(MPPI_500)[0], np.transpose(MPPI_500)[1],label='500 samples')
plt.plot(np.transpose(MPPI_200)[0], np.transpose(MPPI_200)[1],label='200 samples')
plt.plot(np.transpose(MPPI_100)[0], np.transpose(MPPI_100)[1],label='100 samples')
plt.plot(np.transpose(MPPI_50)[0], np.transpose(MPPI_50)[1],label='50 samples')
circle1 = plt.Circle((obstcle_x, obstcle_y), r, color='k', fill=False)
ax = plt.gca()
ax.add_artist(circle1)
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
target_circle = plt.Circle((target[0], target[1]), 0.2, color='b', fill=False)
target_circle_1 = plt.Circle((target[0], target[1]), 0.1, color='b', fill=False)
ax.add_artist(target_circle)
ax.add_artist(target_circle_1)
plt.plot([target[0]], [target[1]], '+', color='b', markersize=50)
ax.grid(True)
ax.set_aspect(1)
plt.xlim([0,4.5])
plt.ylim([0,4.5])
plt.show()

plt.figure(figsize=(8,8))
plt.plot(plot_time(MPPI_500), state_cost(MPPI_500),label='500 samples MPPI')
plt.plot(plot_time(MPPI_200), state_cost(MPPI_200),label='200 samples MPPI')
plt.plot(plot_time(MPPI_100), state_cost(MPPI_100),label='100 samples MPPI')
plt.plot(plot_time(MPPI_50), state_cost(MPPI_50),label='50 samples MPPI')
plt.plot(plot_time(CBF_500), state_cost(CBF_500),label='500 samples MPPI-CBF')
plt.plot(plot_time(CBF_200), state_cost(CBF_200),label='200 samples MPPI-CBF')
plt.plot(plot_time(CBF_100), state_cost(CBF_100),label='100 samples MPPI-CBF')
plt.plot(plot_time(CBF_50), state_cost(CBF_50),label='50 samples MPPI-CBF')
ax = plt.gca()
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
ax.set_xlabel('Time horizon', fontsize=18)
ax.set_ylabel('Costs', fontsize=18)
ax.grid(True)
plt.show()

plt.figure(figsize=(8,8))
plt.plot(plot_time(multi_MPPI_100_40), state_cost(multi_MPPI_100_40),label='100 samples MPPI')
plt.plot(plot_time(multi_MPPI_50_40), state_cost(multi_MPPI_50_40),label='50 samples MPPI')
plt.plot(plot_time(multi_CBF_100_20), state_cost(multi_CBF_100_20),label='100 samples MPPI-CBF')
plt.plot(plot_time(multi_CBF_50_20), state_cost(multi_CBF_50_20),label='50 samples MPPI-CBF')
ax = plt.gca()
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
ax.set_xlabel('Time horizon', fontsize=18)
ax.set_ylabel('Costs', fontsize=18)
ax.grid(True)
plt.show()