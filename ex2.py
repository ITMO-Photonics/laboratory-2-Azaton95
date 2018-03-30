import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def velocity(v,t,g):
    dvdt = g
    return dvdt


t_stop=10.
dt=0.01
t_size=int(t_stop/dt);
v0 = 0
y0 = 20
g = 9.8
e = 0.9
t = np.linspace(0.,t_stop,t_size)
i = np.linspace(0,t.size)

v = odeint(velocity, v0, t, args=(g,))
y = y0 - v * dt;


plt.plot(t,y)
plt.show()
