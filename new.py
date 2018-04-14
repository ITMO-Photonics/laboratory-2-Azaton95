import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def vel(t, vz, g):
    """
    Right hand side of the differential equations
      dvx/dt = -kv*vx
      dvy/dt = -kv*vy-g
    """
    kv=0
    vx, vy = vz
    dvzdt = [-kv*vx, -kv*vy-g]
    return dvzdt

def ball1(x0,y0,v0,alpha,k_res,r,fun):        
    solver = ode(fun)
    solver.set_integrator('dorpi5')
    g = 9.8
    solver.set_f_params(g)
    vx0 = v0*np.sin(np.deg2rad(alpha))
    vy0 = v0*np.cos(np.deg2rad(alpha))
    t0 = 0.0
    vz0 = [vx0, vy0]
    solver.set_initial_value(vz0, t0)
    radius=r/100
    t1 = 40.
    n = 1000
    t = np.linspace(t0, t1, n)
    dt = t[1]
    vz = np.empty((n, 2))
    vx = np.empty_like(t)
    vx[0] = vx0
    vy = np.empty_like(t)
    vy[0] = vy0
    x = np.empty_like(t)
    x[0] = x0
    y = np.empty_like(t)
    y[0] = y0
    vz[0] = vz0
    for i in range(1,n):
        k_res = 1 
        solver.integrate(t[i])
        vz[i] = solver.y
        vx[i] = vz[i,0]
        vy[i] = vz[i,1]
        if x[i-1] + vx[i]*dt <= 0+radius or x[i-1] + vx[i]*dt >= xlim-radius:
            vx[i] = -vx[i]*k_res
        if y[i-1] + vy[i]*dt <= 0+radius or y[i-1] + vy[i]*dt >= ylim-radius:
            vy[i] = -vy[i]*k_res
        t0 = t[i]        
        vz0 = [vx[i], vy[i]]
        solver.set_initial_value(vz0, t0)
        x[i] = x[i-1] + vx[i]*dt
        y[i] = y[i-1] + vy[i]*dt    
    return vx, vy, x, y, x0, y0, r, t
    

def velocity(vz0,t,g):
    kv=0
    vx0,vy0 = vz0
    dvzdt = [-kv*vx0,-kv*vy0-g]
    return dvzdt

def ball2(x0,y0,v0,alpha,k_res,r,fun):
    g=9.8
    radius = r/1000
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
    return vx,vy, x, y, x0, y0, r, t



xlim=10.
ylim=21.
vx, vy, x, y, x0, y0, r, t = ball1(2.,20.,5.,10,1,10,vel)
vx2, vy2, x2, y2, x02, y02, r2, t = ball2(2.,20.,5.,10,1,10,velocity)

m = 1
g = 9.8
W = (m * (np.sqrt(vx ** 2 + vy ** 2)) / 2) + m * g * y
W2 = (m * (np.sqrt(vx2 ** 2 + vy2 ** 2)) / 2) + m * g * y2

fig, ax = plt.subplots()


circle, = ax.plot([], [], 'bo', ms=r)
circle2, = ax.plot([], [], 'go', ms=r2)
coord = np.array([x0,y0])
coord2 = np.array([x02,y02])
k=0

def init():
    ax.set_xlim([0., xlim])
    ax.set_ylim([0., ylim])
    return circle, circle2

def updatefig(frame):
    global k
    coord[0] = x[k]
    coord[1] = y[k]
    circle.set_xdata(coord[0])
    circle.set_ydata(coord[1])
    coord[0] = x2[k]
    coord[1] = y2[k]    
    circle2.set_xdata(coord[0])
    circle2.set_ydata(coord[1])
    k += 1
    return circle, circle2

anim = animation.FuncAnimation(fig, updatefig, frames=y.size, init_func=init, interval=25, blit=True, repeat=False)

plt.show()

plt.plot(t, y, label='y')
plt.plot(t, y2, label='y2')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, x, label='x')
plt.plot(t, x2, label='x2')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, W, label='W')
plt.plot(t, W2, label='W2')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()