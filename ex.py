import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
k=0
def velocity(vz0,t,g):
    vx0,vy0 = vz0
    dvzdt = [-k*vx0,-k*vy0-g]
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

r=10
radius=r/100
xlim=10.
ylim=21.
g=9.8
m=1
c=1
v0=5
alpha=10
vx0 = v0*np.sin(np.deg2rad(alpha))
vy0 = v0*np.cos(np.deg2rad(alpha))
x0 = 2
y0 = 20

vz0 = [vx0, vy0]
z0 = [x0, y0]

t0 = 0.
t_s = 40.
n = 1000
dt = t_s/n
t = np.linspace(t0,t_s,n)
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
    if x[i-1] + vx[i]*dt <= 0+radius or x[i-1] + vx[i]*dt >= xlim-radius:
       vx0 = -vx[i]
       vy0 = vy[i]
       x0 = x[i-1] + vx[i]*dt
       y0 = y[i]
       vz0 = [vx0, vy0]
       z0 = [x0, y0]
       for j in range(i,n):
           tspan = [t[j-1],t[j]]
           vz = odeint(velocity, vz0, tspan, args=(g,))
           vx[j] = vz[1][0]
           vy[j] = vz[1][1]
           vz0 = vz[1]
       x[i] = x[i-1] + vx[i]*dt           
    else:
         x[i] = x[i-1] + vx[i]*dt
    if y[i-1] + vy[i]*dt <= 0+radius or y[i-1] + vy[i]*dt >= ylim-radius:
       vx0 = vx[i]
       vy0 = -vy[i]
       x0 = x[i]
       y0 = y[i-1] + vy[i]*dt
       vz0 = [vx0, vy0]
       z0 = [x0, y0]
       for j in range(i,n):
           tspan = [t[j-1],t[j]]
           vz = odeint(velocity, vz0, tspan, args=(g,))
           vx[j] = vz[1][0]
           vy[j] = vz[1][1]
           vz0 = vz[1]
       y[i] = y[i-1] + vy[i]*dt
    else:
         y[i] = y[i-1] + vy[i]*dt

       

fig, ax = plt.subplots()


circle, = ax.plot([], [], 'bo', ms=r)
coord = np.array([5.,y0])
k=0

def init():
    ax.set_xlim([0., xlim])
    ax.set_ylim([0., ylim])
    return circle,

def updatefig(frame):
    global k
    coord[0] = x[k]
    coord[1] = y[k]
    k += 1
    circle.set_xdata(coord[0])
    circle.set_ydata(coord[1])
    return circle,

anim = animation.FuncAnimation(fig, updatefig, frames=y.size, init_func=init, interval=25, blit=True, repeat=False)

plt.show()
