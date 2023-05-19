import os
import sys
import pandas as pd
import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))
import alphashape
import pickle

with open('LOG_vel_2.pkl', 'rb') as f:
    LOG_vel = pickle.load(f)

success_point, failed_point = [], []
for ii in range(len(LOG_vel)):
    if LOG_vel[ii][0] == 1:
        success_point.append([LOG_vel[ii][1]*0.08-3, LOG_vel[ii][2]*0.08+1])
    else:
        failed_point.append([LOG_vel[ii][1]*0.08-3, LOG_vel[ii][2]*0.08+1])
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
ax.scatter(*zip(*success_point), c='b', s=10)
ax.scatter(*zip(*failed_point), c='r', s=1)
# plt.show()

# alpha_shape = alphashape.alphashape(points_2d, 0.)
# fig, ax = plt.subplots()
# ax.scatter(*zip(*points_2d))                            # plot points
# ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))      # plot alpha shape
# plt.show()

# alpha_shape = alphashape.alphashape(points_2d, 2.0)
# fig, ax = plt.subplots()
# ax.scatter(*zip(*points_2d))
# ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
# plt.show()

x = np.arange(-4,1,0.01)
y = np.sin(0.5 * np.pi * x) +1
plt.axis([-4.0, 1.0, -2.0, 3.0])
plt.plot(x, y, color='k', linewidth=1.2)
plt.plot(x, y-1.5, color='k', linewidth=1.2)
plt.fill_between(x, 3, y, color='whitesmoke')
plt.fill_between(x, y-1.5, -2.0, color='whitesmoke')



plt.show()