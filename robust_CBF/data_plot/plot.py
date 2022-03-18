import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import loadtxt

cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))

# Print the type of the returned object
print("os.getcwd() returns an object of type: {0}".format(type(cwd)))

def sin_plot(dataname):
    trajectories = loadtxt(dataname, delimiter=',')
    plt.plot(trajectories[0], trajectories[1])

def multi_sin_plot():
    sin_plot('robust_CBF\data_plot\A300sample_20steps_sin_CBF_20220318-143623.csv')
    sin_plot('robust_CBF\data_plot\A500sample_20steps_sin_CBF_20220318-104757.csv')
    x = np.arange(0,4,0.01)
    y = np.sin(0.5 * np.pi * x)
    plt.plot(x, y)
    plt.plot(x, y+1)
    plt.show()


def plot_sample(dataname):
    loadedarray = np.load(dataname)
    print(np.shape(loadedarray))
    for i in range(len(loadedarray)):
        sample = loadedarray[i].T
        for j in range(len(sample)):
            trajectory = sample[i]
            plt.plot(trajectory[0], trajectory[1])
    x = np.arange(0,4,0.01)
    y = np.sin(0.5 * np.pi * x)
    plt.plot(x, y)
    plt.plot(x, y+1)
    plt.show()


# plot_sample('robust_CBF/data_plot/B200sample_20steps_sin_MPPI_20220317-172405.npy')
# plot_sample('robust_CBF\data_plot\B200sample_20steps_sin_CBF_20220318-004220.npy')
multi_sin_plot()
