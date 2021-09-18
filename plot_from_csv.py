from numpy import loadtxt
import numpy as np
import math
import matplotlib.pyplot as plt
from mppi_trajectory_follow import MPPI




# load array
data = loadtxt('data.csv', delimiter=',')
# print the array
print(data)

self.t = np.linspace(0,self.tau*self.iter, num=self.iter)
self.X = self.X[0:self.iter]
plt.plot(np.transpose(self.X)[0], np.transpose(self.X)[1],label='x-y')
target_circle = plt.Circle((self.target[0], self.target[1]), 0.2, color='b', fill=False)
ax = plt.gca()
ax.add_artist(target_circle)