import numpy as np
import os

cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))

# Print the type of the returned object
print("os.getcwd() returns an object of type: {0}".format(type(cwd)))

loadedarray = np.load('robust_CBF/data_plot/B200sample_20steps_sin_MPPI_20220317-172405.npy')
print(np.shape(loadedarray))