from numpy import loadtxt
import numpy as np
import math
import matplotlib.pyplot as plt
from mppi_trajectory_follow import MPPI
from cvxopt import matrix


obstcle_x = 2.2
obstcle_y = 2.0
r =0.5
target = [4.0,4.0]

Obstacle_X = matrix([1.5, 1.5, 3.5])
Obstacle_Y = matrix([1.0, 3.3, 2.5])
R = matrix([1.0, 1.0, 1.0])

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
legend = ax.legend(loc='lower right', shadow=True, fontsize='medium')
target_circle = plt.Circle((target[0], target[1]), 0.2, color='b', fill=False)
ax.add_artist(target_circle)
plt.xlim([0,4.5])
plt.ylim([0,4.5])
plt.show()

multi_CBF_100_20 = loadtxt('100sample_multi_CBF.csv', delimiter=',')
multi_CBF_50_20 = loadtxt('50sample_20steps_multi_CBF.csv', delimiter=',')
multi_MPPI_50_40 = loadtxt('50sample_40steps_multi_MPPI.csv', delimiter=',')
multi_MPPI_100_40 = loadtxt('100sample_40steps_multi_MPPI.csv',delimiter=',')

plt.plot(np.transpose(multi_CBF_100_20)[0], np.transpose(multi_CBF_100_20)[1],label='100_samples_MPPI_CBF')
plt.plot(np.transpose(multi_CBF_50_20)[0], np.transpose(multi_CBF_50_20)[1],label='50_samples_MPPI_CBF')
plt.plot(np.transpose(multi_MPPI_100_40)[0], np.transpose(multi_MPPI_100_40)[1],label='100_samples_MPPI')
plt.plot(np.transpose(multi_MPPI_50_40)[0], np.transpose(multi_MPPI_50_40)[1],label='50_samples_MPPI')


for i in range(len(R)):
    circle1 = plt.Circle((Obstacle_X[i], Obstacle_Y[i]), R[i], color='r', fill=False)
    ax = plt.gca()
    ax.add_artist(circle1)
plt.xlim([0,4.5])
plt.ylim([0,4.5])
legend = ax.legend(loc='lower right', shadow=True, fontsize='medium')
target_circle = plt.Circle((target[0], target[1]), 0.2, color='b', fill=False)
ax.add_artist(target_circle)
plt.show()

MPPI_500 = loadtxt('500sample_20steps_single_MPPI.csv',delimiter=',')
MPPI_200 = loadtxt('200sample_20steps_single_MPPI.csv', delimiter=',')
MPPI_100 = loadtxt('100sample_20steps_single_MPPI.csv', delimiter=',')
MPPI_50 = loadtxt('50sample_20steps_single_MPPI.csv', delimiter=',')

plt.plot(np.transpose(MPPI_500)[0], np.transpose(MPPI_500)[1],label='500_samples')
plt.plot(np.transpose(MPPI_200)[0], np.transpose(MPPI_200)[1],label='200_samples')
plt.plot(np.transpose(MPPI_100)[0], np.transpose(MPPI_100)[1],label='100_samples')
plt.plot(np.transpose(MPPI_50)[0], np.transpose(MPPI_50)[1],label='50_samples')
circle1 = plt.Circle((obstcle_x, obstcle_y), r, color='r', fill=False)
ax = plt.gca()
ax.add_artist(circle1)
legend = ax.legend(loc='lower right', shadow=True, fontsize='medium')
target_circle = plt.Circle((target[0], target[1]), 0.2, color='b', fill=False)
ax.add_artist(target_circle)
plt.xlim([0,4.5])
plt.ylim([0,4.5])
plt.show()