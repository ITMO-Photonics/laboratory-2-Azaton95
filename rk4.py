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

 
dvx = RK4(lambda t, vx: 0)
g = -9.8
dvy = RK4(lambda t, vy: -g)

t, vy, dt = 0., 0., .1
while t <= 10:
    if abs(round(t) - t) < 1e-5:
        print("vy(%2.1f)\t= %4.6f " % ( t, vy))
    t, vy = t + dt, vy + dvy( t, vy, dt )

t, vx, dt =0., 2., .1
while t <= 10:
    if abs(round(t) - t) < 1e-5:
        print("vx(%2.1f)\t= %4.6f " % ( t, vx))
    t, vx = t + dt, vx + dvx( t, vx, dt )