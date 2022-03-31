import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import loadtxt
from Unicycle_dynamic import Unicycle_dynamic
import time
import pylab
import torch

cwd = os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(cwd))

dynamic = Unicycle_dynamic()

def sin_plot(dataname, label_name):
    trajectories = loadtxt(dataname, delimiter=',')
    plt.plot(trajectories[0], trajectories[1], label=label_name)

def sin_plot_dot(dataname, label_name):
    trajectories = loadtxt(dataname, delimiter=',')
    plt.plot(trajectories[0], trajectories[1], label=label_name, linestyle='--')

def multi_sin_plot():
   
    sin_plot('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220320-090119.csv','500-sample SCBF-MPPI')
    sin_plot('robust_CBF/data_plot/A400sample_20steps_sin_CBF_20220320-090135.csv', '400-sample SCBF-MPPI')
    sin_plot('robust_CBF/data_plot/A300sample_20steps_sin_CBF_20220320-142155.csv', '300-sample SCBF-MPPI')
    sin_plot('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220320-002409.csv', '200-sample SCBF-MPPI')
   
    sin_plot_dot('robust_CBF/data_plot/A5000sample_20steps_sin_MPPI_20220320-144304.csv', '5000-sample MPPI')
    sin_plot_dot('robust_CBF/data_plot/A4000sample_20steps_sin_MPPI_20220320-144230.csv', '4000-sample MPPI')
    sin_plot_dot('robust_CBF/data_plot/A1000sample_20steps_sin_MPPI_20220320-144201.csv', '1000-sample MPPI')
    sin_plot_dot('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220320-144139.csv', '500-sample MPPI')
    
   
    x = np.arange(0,4.2,0.01)
    y = np.sin(0.5 * np.pi * x)
    plt.plot(x, y, color='k', linewidth=1.2)
    plt.plot(x, y+1, color='k', linewidth=1.2)
    plt.fill_between(x,5, y+1, color='whitesmoke')
    plt.fill_between(x,y, -1.5, color='whitesmoke')

    ax = plt.gca()
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
    plt.xlim((0,4.2))
    plt.ylim((-1.2, 3))
    target_circle = plt.Circle((dynamic.target_pos_x, dynamic.target_pos_y), 0.125, color='b', fill=False)
    target_circle_1 = plt.Circle((dynamic.target_pos_x, dynamic.target_pos_y), 0.075, color='b', fill=False)
    ax.add_artist(target_circle)
    ax.add_artist(target_circle_1)
    plt.plot([dynamic.target_pos_x], [dynamic.target_pos_y], '+', color='b', markersize=30)
    plt.title('Trajecotry of unicycle model')
   
    
    # ax.grid(True)
    ax.set_aspect(0.9)

    ax.set_xlabel

    plt.legend(loc='best', shadow=False, fontsize='small', edgecolor='whitesmoke', facecolor='whitesmoke', ncol=2)
    # plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('robust_CBF/data_plot/multi_sin_{}.eps'.format(timestr), format='eps')
   

    plt.show()  


def plot_sample(dataname):
    loadedarray = np.load(dataname)
    print(np.shape(loadedarray), len(loadedarray))
    for i in range(len(loadedarray)):
        sample = loadedarray[i].T
        for j in range(len(sample)):
            trajectory = sample[j]
            plt.plot(trajectory[0], trajectory[1], color='deepskyblue', linewidth=0.05)
        
    x = np.arange(-1,4.5,0.01)
    y = np.sin(0.5 * np.pi * x)
    plt.plot(x, y, color='k', label='obstacles')
    plt.plot(x, y+1, color='k')
    plt.plot(0,0, color='deepskyblue', label='samples')
    plt.fill_between(x,5, y+1, color='whitesmoke')
    plt.fill_between(x,y, -1.5, color='whitesmoke')
    ax = plt.gca()
    ax.grid(True)
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
    plt.xlim((-1,4.5))
    plt.ylim((-1.2, 2.2))
    plt.legend(loc="upper right")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('robust_CBF/data_plot/multi_sample_{}.eps'.format(timestr), format='eps')
    plt.show()

def plot_sample_single(dataname):
    loadedarray = np.load(dataname)
    print(np.shape(loadedarray), len(loadedarray))
    for i in range(len(loadedarray)):
        sample = loadedarray[i].T
        if i == 20:
            for j in range(len(sample)):
                trajectory = sample[j]
                plt.plot(trajectory[0], trajectory[1], color='deepskyblue', linewidth=0.05)
        
            x = np.arange(0,4,0.01)
            y = np.sin(0.5 * np.pi * x)
            plt.plot(x, y, color='r', label='obstacles')
            plt.plot(x, y+1, color='r')
            ax = plt.gca()
            ax.grid(True)
            ax.set_xlabel('x-position')
            ax.set_ylabel('y-position')
        
            plt.show()

def calculate_cost(data, data_name):
    trajectories = loadtxt(data, delimiter=',')
    costs = (dynamic.target_pos_x - trajectories[0]) ** 2 +\
            (dynamic.target_pos_y - trajectories[1]) ** 2
    time_stpes = len(trajectories.T)
    t = np.linspace(0, time_stpes*dynamic.dt, num=time_stpes)
    plt.plot(t, costs, label=data_name)
    # plt.show()
    # print(costs)

def multi_cost():
    calculate_cost('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220320-090119.csv','500-sample SCBF-MPPI')
    calculate_cost('robust_CBF/data_plot/A400sample_20steps_sin_CBF_20220320-090135.csv', '400-samples SCBF-MPPI')
    calculate_cost('robust_CBF/data_plot/A300sample_20steps_sin_CBF_20220320-142155.csv', '300-samples SCBF-MPPI')
    calculate_cost('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220320-002409.csv', '200-samples SCBF-MPPI')

    calculate_cost('robust_CBF/data_plot/A5000sample_20steps_sin_MPPI_20220320-144304.csv', '5000-samples MPPI')
    calculate_cost('robust_CBF/data_plot/A4000sample_20steps_sin_MPPI_20220320-144230.csv', '4000-samples MPPI')
    calculate_cost('robust_CBF/data_plot/A1000sample_20steps_sin_MPPI_20220320-144201.csv', '1000-samples MPPI')
    calculate_cost('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220320-144139.csv', '500-samples MPPI')
    
    ax = plt.gca()
    ax.grid(True)
    legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
    ax.set_xlabel('Time Horizon')
    ax.set_ylabel('Costs')
   
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('robust_CBF/data_plot/multi_cost_{}.eps'.format(timestr), format='eps')
    plt.show()

def variance(data):
     # Number of observations
     n = len(data)
     # Mean of the data
     mean = sum(data) / n
     # Square deviations
     deviations = [(x - mean) ** 2 for x in data]
     # Variance
     variance = sum(deviations) / n
     return variance

def check_control(data, data_name):
    control = loadtxt(data, delimiter=',')
    time_stpes = len(control.T)
    # print(time_stpes)
    t = np.linspace(0, time_stpes*dynamic.dt, num=time_stpes)
    plt.plot(t, control[1], label=data_name)
    plt.plot(t, control[2])
    print(variance(control[1]))
    plt.show()

def table_data(data):
    trajectory = loadtxt(data, delimiter=',').T
    time_steps = len(trajectory)
    print(time_steps)
    collision_time = 0
    for i in range(time_steps):
        x = trajectory[i][0]
        y = trajectory[i][1]
        # print(x,y)
        if y > np.sin(0.5*np.pi*x)+1 or y < np.sin(0.5*np.pi*x):
            collision_time += 1
    collision_rate = collision_time/time_steps
    return collision_rate, time_steps

def multi_table():
    
    a1, b1 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220320-090119.csv')
    a2, b2 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220324-053201.csv')
    a3, b3 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220323-112330.csv')
    a4, b4 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220323-112441.csv')
    a5, b5 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220323-162850.csv')
    a6, b6 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220323-210943.csv')
    a7, b7 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220323-223905.csv')
    a8, b8 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220324-032131.csv')
    a9, b9 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220324-123336.csv')
    a10, b10 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220324-181525.csv')
    a = (a1+a2+a3+a4+a5+a6+a7+a8+a9+a10)/10
    b = (b1+b2+b3+b4+b5+b6+b7+b8+b9+b10)/10
    # ('robust_CBF/data_plot/A5000sample_40steps_sin_MPPI_20220320-144603.csv')
    print(a, b)

    c1, d1 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220317-222242.csv')
    c2, d2 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220318-004220.csv')
    c3, d3 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220320-002409.csv')
    c4, d4 = table_data('robust_CBF/data_plot/A100sample_20steps_sin_CBF_20220318-022419.csv')
    c5, d5 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220325-022139.csv')
    c6, d6 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220325-022338.csv')
    c7, d7 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220325-094724.csv')
    c8, d8 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220325-094749.csv')
    c9, d9 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220325-122825.csv')
    c10, d10 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220325-152440.csv')
    c = (c1+c2+c3+c4+c5+c6+c7+c8+c9+c10)/10
    d = (d1+d2+d3+d4+d5+d6+d7+d8+d9+d10)/10
    print(c,d)

    # C1, D1 = table_data('robust_CBF/data_plot/B200sample_20steps_sin_MPPI_20220325-203909.npy')
    C2, D2 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-203909.csv')
    C3, D3 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-210459.csv')
    C4, D4 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-210359.csv')
    C5, D5 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-203940.csv')
    C6, D6 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-203946.csv')
    C7, D7 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-210244.csv')
    C8, D8 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-210134.csv')
    C9, D9 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-210035.csv')
    C10, D10 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-205844.csv')
    C1, D1 = table_data('robust_CBF/data_plot/A200sample_20steps_sin_MPPI_20220325-205420.csv')
    # C1, D1 = table_data('robust_CBF/data_plot/B200sample_20steps_sin_MPPI_20220325-203957.csv')
    C = (C1+C2+C3+C4+C5+C6+C7+C8+C9+C10)/10
    D = (D1+D2+D3+D4+D5+D6+D7+D8+D9+D10)/10
    print(C,D)

    A1, B1 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220325-210743.csv')
    A2, B2 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220325-211237.csv')
    A3, B3 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220325-211322.csv')
    A4, B4 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220325-211350.csv')
    A5, B5 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220325-211545.csv')
    A6, B6 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220325-211609.csv')
    A7, B7 = table_data('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220325-211627.csv')
    A8, B8 = table_data('robust_CBF/data_plot/C500sample_20steps_sin_MPPI_20220325-211809.csv')
    A9, B9 = table_data('robust_CBF/data_plot/C500sample_20steps_sin_MPPI_20220325-211703.csv')
    A10, B10 = table_data('robust_CBF/data_plot/C500sample_20steps_sin_MPPI_20220325-211814.csv')
    A = (A1+A2+A3+A4+A5+A6+A7+A8+A9+A10)/10
    B = (B1+B2+B3+B4+B5+B6+B7+B8+B9+B10)/10
    print(A, B)


def calculate_N(dataname):
    epsilon_1, rho_1 = 0.05, 0.05
    epsilon_2, rho_2 = 0.1, 0.1
    loadedarray = np.load(dataname, allow_pickle=True)
    print(np.shape(loadedarray), len(loadedarray))
    E1, ES = [], []
    for i in range(len(loadedarray)):
        sample = loadedarray[i]
        # print(sample)
        E_1 = (torch.sum(torch.exp(-sample)))#/dynamic.K
        print(0.1-E_1, i)
        E_S = (torch.sum(sample))/dynamic.K
        E1.append(E_1)
        ES.append(E_S)
        # print(E_S)

    N_1 = -1/(epsilon_1**2) *(np.log(rho_1/2))
    print(N_1)
    N_2 = (4/(rho_2*epsilon_2**2)) * 100 # ((1/(E1[30] - epsilon_1))**2)
    print(N_2)

        
# calculate_N('robust_CBF/data_plot/B500sample_20steps_sin_CBF_20220328-210401.npy')
# calculate_N('robust_CBF/data_plot/D500sample_20steps_sin_MPPI_20220328-172240.npy')
# plot_sample('robust_CBF/data_plot/B200sample_20steps_sin_MPPI_20220317-172405.npy')
# plot_sample('robust_CBF/data_plot/B200sample_20steps_sin_CBF_20220318-004220.npy')
# plot_sample('robust_CBF/data_plot/B200sample_20steps_sin_CBF_20220320-002409.npy')
multi_sin_plot()
multi_cost()
# check_control('robust_CBF/data_plot/C500sample_20steps_sin_CBF_20220324-032131.csv', 'control_MPPI_CBF_500_sample')
# check_control('robust_CBF/data_plot/C500sample_20steps_sin_CBF_20220323-210943.csv', 'control_MPPI_CBF_500_sample')
# check_control('robust_CBF/data_plot/C200sample_20steps_sin_MPPI_20220324-235015.csv', 'control_MPPI_200_sample')
# plot_sample_single('robust_CBF/data_plot/B200sample_20steps_sin_CBF_20220318-004220.npy')
# plot_sample_single('robust_CBF/data_plot/B500sample_20steps_sin_CBF_20220324-032131.npy')
# multi_table()





