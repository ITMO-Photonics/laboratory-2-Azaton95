import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3   ) )
	    )( dt * f( t + dt/2, y + dy2/2 ) )
	    )( dt * f( t + dt/2, y + dy1/2 ) )
	    )( dt * f( t       , y         ) )

 
def ballrk4(x0,y0,v0,alpha,k_res,r,n):
    dvx = RK4(lambda t, vx: 0)
    g = 9.8
    dvy = RK4(lambda t, vy: -g)
    vx0 = v0*np.sin(np.deg2rad(alpha))
    vy0 = v0*np.cos(np.deg2rad(alpha))
    t0 = 0.0
    t1 = 40.0
    radius=r/100
    t = np.linspace(t0, t1, n)
    dt = t[1]
    vx = np.empty_like(t)
    vy = np.empty_like(t)
    vx[0] = vx0
    vy[0] = vy0
    x = np.empty_like(t)
    y = np.empty_like(t)
    x[0] = x0
    y[0] = y0
    g = 9.8
    vxf = vx0
    vyf = vy0
    for i in range(1,n):
        vy[i] = vy[i-1] + dvy( t[i-1], vy[i-1], dt )
        if y[i-1] + vy[i]*dt <= 0+radius or y[i-1] + vy[i]*dt >= ylim-radius:
            vy[i] = -vy[i]*k_res 
            vyf = -vyf    
        vx[i] = vx[i-1] + dvx( t[i-1], vx[i-1], dt )
        if x[i-1] + vx[i]*dt <= 0+radius or x[i-1] + vx[i]*dt >= xlim-radius:
            vx[i] = -vx[i]*k_res
            vxf = -vxf
        x[i] = x[i-1] + vx[i]*dt
        y[i] = y[i-1] + vy[i]*dt   
    return vx, vy, x, y, x0, y0, r, n, t


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

def vxf(t, vx, g):
    kv=0    
    dvxdt = -kv*vx
    return dvxdt

def vyf(t,vy,g):
    kv=0    
    dvydt = -kv*vy-g
    return dvydt


def Eul(x0,y0,v0,alpha,k_res,r,n,fun1,fun2):
    vx0 = v0*np.sin(np.deg2rad(alpha))
    vy0 = v0*np.cos(np.deg2rad(alpha))
    t0 = 0.0
    t1 = 40.0
    radius=r/100
    t = np.linspace(t0, t1, n)
    dt = t[1]
    vx = np.empty_like(t)
    vy = np.empty_like(t)
    vx[0] = vx0
    vy[0] = vy0
    x = np.empty_like(t)
    y = np.empty_like(t)
    x[0] = x0
    y[0] = y0
    g = 9.8
    vxf = vx0
    vyf = vy0
    for i in range(1,n):
        vxf += fun1(dt,vx[i-1],g)*dt
        vyf += fun2(dt,vy[i-1],g)*dt
        vx[i] = vxf
        vy[i] = vyf
        if x[i-1] + vx[i]*dt <= 0+radius or x[i-1] + vx[i]*dt >= xlim-radius:
            vx[i] = -vx[i]*k_res
            vxf = -vxf
        if y[i-1] + vy[i]*dt <= 0+radius or y[i-1] + vy[i]*dt >= ylim-radius:
            vy[i] = -vy[i]*k_res 
            vyf = -vyf
        x[i] = x[i-1] + vx[i]*dt
        y[i] = y[i-1] + vy[i]*dt   
    return vx, vy, x, y, x0, y0, r, n, t

def ball1(x0,y0,v0,alpha,k_res,r,n,method,fun):        
    solver = ode(fun)
    solver.set_integrator(method)
    g = 9.8
    solver.set_f_params(g)
    vx0 = v0*np.sin(np.deg2rad(alpha))
    vy0 = v0*np.cos(np.deg2rad(alpha))
    t0 = 0.0
    vz0 = [vx0, vy0]
    solver.set_initial_value(vz0, t0)
    radius=r/100
    t1 = 40.
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
    return vx, vy, x, y, x0, y0, r, n, t
    

def velocity(vz0,t,g):
    kv=0
    vx0,vy0 = vz0
    dvzdt = [-kv*vx0,-kv*vy0-g]
    return dvzdt

def ball2(x0,y0,v0,alpha,k_res,r,n,fun):
    g=9.8
    radius = r/1000
    vx0, vy0 = v0*np.sin(np.deg2rad(alpha)), v0*np.cos(np.deg2rad(alpha))
    vz0 = [vx0, vy0]
    t0 = 0.
    t_s = 40.
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
    return vx,vy, x, y, x0, y0, r, n, t



xlim=10.
ylim=21.
vx, vy, x, y, x0, y0, r, n, t = ball1(2.,20.,5.,10,1,10,10000,'dopri5',vel)
#vx2, vy2, x2, y2, x02, y02, r2, n, t = ball2(2.,20.,5.,10,1,10,10000,velocity)
#vx2, vy2, x2, y2, x02, y02, r2, n, t = Eul(2.,20.,5.,10,1,9,10000,vxf,vyf)
vx2, vy2, x2, y2, x02, y02, r2, n, t = ballrk4(2.,20.,5.,10,1,9,10000)

m = 1
g = 9.8

W_k = (m * (np.sqrt(vx ** 2 + vy ** 2)) ** 2 / 2) 
W_k2 = (m * (np.sqrt(vx2 ** 2 + vy2 ** 2)) ** 2 / 2)

W_p = m * g * y
W_p2 = m * g * y2

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

anim = animation.FuncAnimation(fig, updatefig, frames=y.size, init_func=init, interval=5*(10000/n), blit=True, repeat=False)

plt.show()

plt.plot(t, vy, label='vy')
plt.plot(t, vy2, label='vy2')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, vx, label='vx')
plt.plot(t, vx2, label='vx2')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, W_k, label='W_k')
plt.plot(t, W_p, label='W_p')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, W_k, label='W_k2')
plt.plot(t, W_p, label='W_p2')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()