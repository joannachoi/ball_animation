import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as M


#------------------------------
# set global variables
N = 50
R = 0.05
m = 2
a = np.array([0.0, -9.8])
dt = 0.05
time = np.arange(0, 1000, dt)
fig, ax = plt.subplots()
ax.set_xlim([-1, 1])
ax.set_ylim([-1,1])
particles, = ax.plot([],[])
r = np.zeros((2, N))
r_xval = (2-2*R)*np.random.rand(N) - (1-R)
r_yval = (2-2*R)*np.random.rand(N) - (1-R)
for i in range(N):
    r[0][i] = r_xval[i]
    r[1][i] = r_yval[i]
v = np.zeros((2, N))
rand_1 = np.random.rand(N)
rand_2 = np.random.rand(N)
for i in range(N):
    v[0][i] = rand_1[i]*np.cos(2*np.pi*rand_2[i])
    v[1][i] = rand_1[i]*np.sin(2*np.pi*rand_2[i])
pressure_right = 0
pressure_left = 0
pressure_top = 0
pressure_bottom = 0
#------------------------------


# equations of motion
def advance(r, v, a, dt):
    r_new = r + (v*dt) + (1/2)*a*(dt**2)
    v_new = v + a*dt
    return r_new, v_new


# figure out bounces
def update(r, v, dt):
    global pressure_left, pressure_right, pressure_top, pressure_bottom
    r_new, v_new = advance(r, v, a, dt)
    rightbound = 1-R
    leftbound = -1+R
    upperbound = 1-R
    lowerbound = -1+R
    # test for boundaries
    #---------------------------
    # x pos right
    if r_new[0] > rightbound:
        dt_r = (r_new[0] - rightbound)/v_new[0]
        r2, v2 = advance(r_new, v_new, a, -dt_r)
        v2[0] = -v2[0]
        r3, v3 = advance(r2, v2, a, dt_r)
        pressure_left += 2*abs(v2[0])
        return r3, v3
    # x pos left
    if r_new[0] < leftbound:
        dt_r = (r_new[0] - leftbound)/v_new[0]
        r2, v2 = advance(r_new, v_new, a, -dt_r)
        v2[0] = -v2[0]
        r3, v3 = advance(r2, v2, a, dt_r)
        pressure_right += 2*abs(v2[0])
        return r3, v3
    # y pos up
    if r_new[1] > upperbound:
        quad_num = M.sqrt(v[1]**2 - (4*0.5*a[1]*(r[1]-upperbound)))
        quad_denom = (2*.5*a[1])
        dt_r1 = (-v[1] + quad_num)/quad_denom
        dt_r2 = (-v[1] - quad_num)/quad_denom
        if abs(dt_r1) < abs(dt_r2):
            dt_r = dt_r1
        else:
            dt_r = dt_r2
        r2, v2 = advance(r_new, v_new, a, dt_r)
        v2[1] = -v2[1]
        r3, v3 = advance(r2, v2, a, dt_r)
        pressure_top += 2*abs(v2[1])
        return r3, v3
    # y pos down
    if r_new[1] < lowerbound:
        quad_num = M.sqrt(v_new[1]**2 - (4*0.5*a[1]*(r_new[1]-lowerbound)))
        quad_denom = (2*.5*a[1])
        dt_r1 = (-v_new[1] + quad_num)/quad_denom
        dt_r2 = (-v_new[1] - quad_num)/quad_denom
        if abs(dt_r1) < abs(dt_r2):
            dt_r = dt_r1
        else:
            dt_r = dt_r2
        r2, v2 = advance(r_new, v_new, a, dt_r)
        v2[1] = -v2[1]
        r3, v3 = advance(r2, v2, a, -dt_r)
        pressure_bottom += 2*abs(v2[1])
        return r3, v3
    #--------------------------
    return r_new, v_new


def animate(i):
    global r, v
    r_next = np.zeros((2, N))
    v_next = np.zeros((2, N))
    for i in range(N):
        r_next[:,i], v_next[:,i] = update(r[:,i], v[:,i], dt)
    particles.set_data(r_next[0][:], r_next[1][:])
    r = r_next
    v = v_next
    return particles,


def initialize():
    global particles
    markersize = int(fig.dpi * R * fig.get_figwidth() / np.diff(ax.get_xbound())[0])
    particles, = ax.plot([], [], 'ro', ms=markersize)
    return particles,


def main():
    for ball in range(N):
        ani = animation.FuncAnimation(fig, animate, interval=10, init_func=initialize, blit=False)
    plt.show()
    time = i*dt
    print("Time: " + str(time))
    KE = 0.5*sum(sum(v**2))
    print("Kinetic Energy: " + str(KE))
    print("Pressure on right wall: " + str(pressure_right/time))
    print("Pressure on left wall: " + str(pressure_left/time))
    print("Pressure on top wall: " + str(pressure_top/time))
    print("Pressure on bottom wall: " + str(pressure_bottom/time))


if __name__ == '__main__':
    main()
