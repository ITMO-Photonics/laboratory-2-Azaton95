import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def velocity(vz,t,g):
    vx = vz[0]
    vy = vz[1]
    dvxdt = -k*vx
    dvydt = -k*vy-g
    dvzdt = [dvxdt,dvydt]
    return dvzdt

def position(z,t,i):
    x = z[0]
    y = z[1]
    dxdt = vx[i]
    dydt = vy[i]
    dzdt = [dxdt,dydt]
    return dzdt

def f(y):
    if y>=0:
       f = 0
    else:
       f = -m*c*y
    return f

v0=10
g=9.8
k=0.1
m=1
c=0.9
alpha=90
vx0 = v0*np.sin(np.deg2rad(alpha))
vy0 = v0*np.cos(np.deg2rad(alpha))
x0 = 0
y0 = 20

vz0 = [vx0, vy0]
z0 = [x0, y0]

t_s = 40.
n = 1000
dt = t_s/n
t = np.linspace(0.,t_s,n)
vx = np.empty_like(t)
vy = np.empty_like(t)
x = np.empty_like(t)
y = np.empty_like(t)
vx[0] = vz0[0]
vy[1] = vz0[1]
x[0] = z0[0]
y[0] = z0[1]

for i in range(1,n):
    tspan = [t[i-1],t[i]]
    vz = odeint(velocity, vz0, tspan, args=(g,))
    vx[i] = vz[1][0]
    vy[i] = vz[1][1]
    vz0 = vz[1]

#for i in range(1,n):
#    tspan = [t[i-1],t[i]]
#    z = odeint(position, z0, tspan, args=(i,))
#    x[i] = z[1][0]
#    y[i] = z[1][1]
#    z0 = z[1]

for i in range(1,n):
    x[i] = x[i-1] + vx[i]*dt
    if y[i-1] + vy[i]*dt <= 0:
       y[i]=0
#       vy[i:] = vy[i:] * (-c)
    else:
       y[i] = y[i-1] + vy[i]*dt

       

fig, ax = plt.subplots()

circle, = ax.plot([], [], 'bo', ms=10)
coord = np.array([5.,y0])
k=0

def init():
    ax.set_xlim([0., 40.])
    ax.set_ylim([0., 40.])
    return circle,

def updatefig(frame):
    global k
    coord[0] = x[k]
    coord[1] = y[k]
    k += 1
    circle.set_xdata(coord[0])
    circle.set_ydata(coord[1])
    return circle,

anim = animation.FuncAnimation(fig, updatefig, frames=y.size, init_func=init, interval=100, blit=True, repeat=False)

plt.show()
