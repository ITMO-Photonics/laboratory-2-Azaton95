import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def velocity(v,t,g):
    dvdt = g
    return dvdt


t_stop=10.
dt=0.1
t_size=int(t_stop/dt);
v0 = 0
y0 = 20.
g = 9.8
e = 0.9
t = np.linspace(0.,t_stop,t_size)
i = np.linspace(0,t.size)
y=np.zeros(t.size)

v = odeint(velocity, v0, t, args=(g,))

y[0]=y0
i=1

while i < t.size:
      if y[i] >= 0 and y[i] <= y0:
         y[i]=y[i-1]-v[i,0]*dt
      
      if y[i] < 0:
         y[i] = 0
         v[i:,0] = v[i:,0] * (-e) 
      if y[i] > y0:
         y[i] = y0
         v[i:,0] = v[i:,0] * (-e)
      i += 1
 

fig, ax = plt.subplots()

circle, = ax.plot([], [], 'bo', ms=10)
coord = np.array([5.,y0])
k=0

def init():
    ax.set_xlim([0., 20.])
    ax.set_ylim([0., 20.])
    return circle,

def updatefig(frame):
    global k
    coord[1] = y[k]
    k += 1
    circle.set_xdata(coord[0])
    circle.set_ydata(coord[1])
    return circle,

anim = animation.FuncAnimation(fig, updatefig, frames=y.size, init_func=init, interval=100, blit=True, repeat=False)

plt.show()
