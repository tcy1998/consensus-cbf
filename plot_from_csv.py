from numpy import loadtxt
import numpy as np
import math
import matplotlib.pyplot as plt
from mppi_trajectory_follow import MPPI


obstcle_x = 2.2
obstcle_y = 2.0
r =0.5
target = [4.0,4.0]

# load array
data = loadtxt('data.csv', delimiter=',')

CBF_500 = loadtxt('500sample_single_CBF.csv',delimiter=',')
CBF_200 = loadtxt('200sample_single_CBF.csv', delimiter=',')
CBF_100 = loadtxt('100sample_single_CBF.csv', delimiter=',')
CBF_50 = loadtxt('50sample_single_CBF.csv', delimiter=',')

plt.plot(np.transpose(CBF_500)[0], np.transpose(CBF_500)[1],label='500_samples')
plt.plot(np.transpose(CBF_200)[0], np.transpose(CBF_200)[1],label='200_samples')
plt.plot(np.transpose(CBF_100)[0], np.transpose(CBF_100)[1],label='100_samples')
plt.plot(np.transpose(CBF_50)[0], np.transpose(CBF_50)[1],label='50_samples')
circle1 = plt.Circle((obstcle_x, obstcle_y), r, color='r', fill=False)
ax = plt.gca()
ax.add_artist(circle1)
legend = ax.legend(loc='lower right', shadow=True, fontsize='large')
target_circle = plt.Circle((target[0], target[1]), 0.2, color='b', fill=False)
ax.add_artist(target_circle)
plt.xlim([0,4.5])
plt.ylim([0,4.5])
plt.show()