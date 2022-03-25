import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import loadtxt
from Unicycle_dynamic import Unicycle_dynamic
import time
import pylab

cwd = os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(cwd))

dynamic = Unicycle_dynamic()

def sin_plot(dataname, label_name):
    trajectories = loadtxt(dataname, delimiter=',')
    plt.plot(trajectories[0], trajectories[1], label=label_name)

def multi_sin_plot():
    sin_plot('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220320-002409.csv', '200 samples MPPI-CBF')
    sin_plot('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220320-090119.csv','500 samples MPPI-CBF')
    # sin_plot('robust_CBF/data_plot/A400sample_20steps_sin_CBF_20220320-090135.csv', '400 samples MPPI-CBF')
    # sin_plot('robust_CBF/data_plot/A300sample_20steps_sin_CBF_20220320-142155.csv', '300 samples MPPI-CBF')
   

    sin_plot('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220320-144139.csv', '500 samples MPPI')
    sin_plot('robust_CBF/data_plot/A1000sample_20steps_sin_MPPI_20220320-144201.csv', '1000 samples MPPI')
    # sin_plot('robust_CBF/data_plot/A4000sample_20steps_sin_MPPI_20220320-144230.csv', '4000 samples MPPI')
    x = np.arange(0,4,0.01)
    y = np.sin(0.5 * np.pi * x)
    plt.plot(x, y, color='r')
    plt.plot(x, y+1, color='r')

    ax = plt.gca()
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
    target_circle = plt.Circle((dynamic.target_pos_x, dynamic.target_pos_y), 0.15, color='b', fill=False)
    target_circle_1 = plt.Circle((dynamic.target_pos_x, dynamic.target_pos_y), 0.1, color='b', fill=False)
    ax.add_artist(target_circle)
    ax.add_artist(target_circle_1)
    plt.plot([dynamic.target_pos_x], [dynamic.target_pos_y], '+', color='b', markersize=50)
   
    
    ax.grid(True)
    ax.set_aspect(1)

    ax.set_xlabel

    legend = ax.legend(loc='lower left', shadow=True, fontsize='medium')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # plt.savefig('robust_CBF/data_plot/multi_sin_{}.eps'.format(timestr), format='eps')
   

    plt.show()  


def plot_sample(dataname):
    loadedarray = np.load(dataname)
    print(np.shape(loadedarray), len(loadedarray))
    for i in range(len(loadedarray)):
        sample = loadedarray[i].T
        for j in range(len(sample)):
            trajectory = sample[j]
            plt.plot(trajectory[0], trajectory[1], color='deepskyblue', linewidth=0.05)
        
    x = np.arange(0,4,0.01)
    y = np.sin(0.5 * np.pi * x)
    plt.plot(x, y, color='r', label='obstacles')
    plt.plot(x, y+1, color='r')
    plt.plot(0,0, color='deepskyblue', label='samples')
    ax = plt.gca()
    ax.grid(True)
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
    plt.legend(loc="upper right")
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
    calculate_cost('robust_CBF/data_plot/A500sample_20steps_sin_CBF_20220320-090119.csv','500 samples MPPI-CBF')
    calculate_cost('robust_CBF/data_plot/A400sample_20steps_sin_CBF_20220320-090135.csv', '400 samples MPPI-CBF')
    calculate_cost('robust_CBF/data_plot/A300sample_20steps_sin_CBF_20220320-142155.csv', '300 samples MPPI-CBF')
    calculate_cost('robust_CBF/data_plot/A200sample_20steps_sin_CBF_20220320-002409.csv', '200 samples MPPI-CBF')
    calculate_cost('robust_CBF/data_plot/A4000sample_20steps_sin_MPPI_20220320-144230.csv', '4000 samples MPPI')
    calculate_cost('robust_CBF/data_plot/A1000sample_20steps_sin_MPPI_20220320-144201.csv', '1000 samples MPPI')
    calculate_cost('robust_CBF/data_plot/A500sample_20steps_sin_MPPI_20220320-144139.csv', '500 samples MPPI')
    
    ax = plt.gca()
    ax.grid(True)
    legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
   
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('robust_CBF/data_plot/multi_cost_{}.eps'.format(timestr), format='eps')
    plt.show()

# def compute_error(data, data_name):



# plot_sample('robust_CBF/data_plot/B200sample_20steps_sin_MPPI_20220317-172405.npy')
# plot_sample('robust_CBF/data_plot/B200sample_20steps_sin_CBF_20220318-004220.npy')
# plot_sample('robust_CBF/data_plot/B200sample_20steps_sin_CBF_20220320-002409.npy')
# multi_sin_plot()
multi_cost()

# plot_sample_single('robust_CBF/data_plot/B200sample_20steps_sin_CBF_20220318-004220.npy')
# plot_sample_single('robust_CBF/data_plot/B500sample_20steps_sin_CBF_20220324-032131.npy')
