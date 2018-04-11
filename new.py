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
    f = [-kv*vx, -kv*vy-g]
    return f

# Create an `ode` instance to solve the system of differential
# equations defined by `vel`, and set the solver method to 'dop853'.
solver = ode(vel)
solver.set_integrator('dop853')

# Give the value of omega to the solver. This is passed to
# `vel` when the solver calls it.
g = 9.8
solver.set_f_params(g)

# Set the initial value z(0) = z0.
v0 = 5
alpha = 10
vx0 = v0*np.sin(np.deg2rad(alpha))
vy0 = v0*np.cos(np.deg2rad(alpha))
t0 = 0.0
vz0 = [vx0, vy0]
solver.set_initial_value(vz0, t0)

# Create the array `t` of time values at which to compute
# the solution, and create an array to hold the solution.
# Put the initial value in the solution array.
x0 = 2.0
y0 = 20.0
r=10
radius=r/100
xlim=10.
ylim=21.
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

# Repeatedly call the `integrate` method to advance the
# solution to time t[k], and save the solution in vz[k].
k = 1
while solver.successful() and solver.t < t1:
    solver.integrate(t[k])
    vz[k] = solver.y
    vx[k] = vz[k,0]
    vy[k] = vz[k,1]
    k += 1


for i in range(1,n):
    if x[i-1] + vx[i]*dt <= 0+radius or x[i-1] + vx[i]*dt >= xlim-radius:
       vx0 = -vx[i]
       vy0 = vy[i]
       x0 = x[i-1] + vx[i]*dt
       y0 = y[i]
       t0 = t[i]
       vz0 = [vx0, vy0]
       solver.set_initial_value(vz0, t0)
       for j in range(i,n):
             solver.integrate(t[j])
             vz[j] = solver.y
             vx[j] = vz[j,0]
             vy[j] = vz[j,1]             
       x[i] = x[i-1] + vx[i]*dt
    else:
         x[i] = x[i-1] + vx[i]*dt
    if y[i-1] + vy[i]*dt <= 0+radius or y[i-1] + vy[i]*dt >= ylim-radius:
       vx0 = vx[i]
       vy0 = -vy[i]
       x0 = x[i]
       y0 = y[i-1] + vy[i]*dt
       t0 = t[i]
       vz0 = [vx0, vy0]
       solver.set_initial_value(vz0, t0)
       k = i
       for j in range(i,n):
             solver.integrate(t[j])
             vz[j] = solver.y
             vx[j] = vz[j,0]
             vy[j] = vz[j,1]             
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

