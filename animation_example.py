import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

complex_vec = np.arange(5,6,.001)
real_vec = np.arange(7,8,.001)
time_vec = np.arange(0,1,.001)
num_files = np.size(time_vec)

#creating the modulus vector
modulus_vec = np.zeros(np.shape(complex_vec))
print(complex_vec)
for k in range (0,complex_vec.size):
    a = complex_vec[k]
    b = real_vec[k]
    calc_modulus = np.sqrt(a**2 + b**2)
    modulus_vec[k] = calc_modulus

#finally animateing
fig = plt.figure()
ax = plt.axes(xlim = (-1,1) ,ylim = (-1,15))#limits were     arbitrary
#line = ax.plot([],[])
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = time_vec[i]
    y = complex_vec[i]
    y1 = real_vec[i] 
    y2 = modulus_vec[i]
    # notice we are only calling set_data once, and bundling the y values into an array
    line.set_data(x,np.array([y, y1, y2]))
    print(x,np.array([y, y1, y2]))
    return line,

animation_object = animation.FuncAnimation(fig, 
                                           animate, 
                                           init_func= init, 
                                           frames = num_files,
                                           interval = 30, 
                                           blit = True)

#turnn this line on to save as mp4
#anim.save("give it a name.mp4", fps = 30, extra-args = ['vcodec',     'libx264'])
plt.show()