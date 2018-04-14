import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def velocity(vz0,t,g):
    kv=0
    vx0,vy0 = vz0
    dvzdt = [-kv*vx0,-kv*vy0-g]
    return dvzdt

def ball(x0,y0,v0,alpha,fun):
    g=9.8
    k_res=1
    vx0, vy0 = v0*np.sin(np.deg2rad(alpha)), v0*np.cos(np.deg2rad(alpha))
    vz0 = [vx0, vy0]
    t0 = 0.
    t_s = 40.
    n = 1000
    t = np.linspace(t0,t_s,n)
    dt = t[1]
    vx, vy, x, y = np.empty_like(t), np.empty_like(t), np.empty_like(t), np.empty_like(t)
    x[0], y[0] = x0, y0

    vx[0], vy[0] = vz0

    for i in range(1,n):
        tspan = [t[i-1],t[i]]
        vz = odeint(fun, vz0, tspan, args=(g,))
        vx[i] = vz[1][0]
        vy[i] = vz[1][1]
        if x[i-1] + vx[i]*dt <= 0+radius or x[i-1] + vx[i]*dt >= xlim-radius:
            vx[i] = -vx[i]*k_res
        if y[i-1] + vy[i]*dt <= 0+radius or y[i-1] + vy[i]*dt >= ylim-radius:
            vy[i] = -vy[i]*k_res        
        vz0 = [vx[i], vy[i]]
        x[i] = x[i-1] + vx[i]*dt
        y[i] = y[i-1] + vy[i]*dt 

    return x, y, x0, y0


r=10
radius=r/100
xlim=10.
ylim=21.
x,y,x0,y0 = ball(2.,20.,5.,40,velocity)
fig, ax = plt.subplots()
circle, = ax.plot([], [], 'bo', ms=r)
coord = np.array([x0,y0])

def init():
    ax.set_xlim([0., xlim])
    ax.set_ylim([0., ylim])
    return circle,
k=0
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
