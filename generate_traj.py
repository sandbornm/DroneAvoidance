from casadi import *
import numpy as np
import matplotlib.pyplot as plt


'''
generate_traj
Inputs:
x_start: 3 element list of starting/current position of x
goal: 3 element list of destination
N: Number of time steps
cx: coefficient to perturb midpoint of path in the x direction
cy: ... in the y direction
cz: ... in the z direction

Outputs:
(N, 8) dimensional array
In this order from index 0 - 7
x
y
z
phi - yaw
vx
vy
vz
vphi - yaw-rate

Examples of use cases are shown below
'''
def generate_traj(x_start, goal, N, dt, cx, cy, cz):
    k_x = 1
    k_y = 1
    k_z = 1
    k_phi = pi / 180
    tau_x = 0.8355
    tau_y = 0.7701
    tau_z = 0.5013
    tau_phi = 0.5142
    nx = 8
    nu = 4

    # ---- dynamic constraints --------
    def f(b, w):
        ret = []
        ret.append(b[4] * np.cos(b[3]) - b[5] * np.sin(b[3]))
        ret.append(b[4] * sin(b[3]) + b[5] * cos(b[3]))
        ret.append(b[6])
        ret.append(b[7])
        ret.append((-b[4] + k_x * w[0]) / tau_x)
        ret.append((-b[5] + k_y * w[1]) / tau_y)
        ret.append((-b[6] + k_z * w[2]) / tau_z)
        ret.append((-b[7] + k_phi * w[3]) / tau_phi)
        return vcat(ret)

    opti = Opti()  # Optimization problem

    # ---- decision variables ---------
    X = opti.variable(nx, N + 1)  # state trajectory
    pos = X[0:3, :]
    speed = X[4:7, :]
    midpoint = (np.array(x_start) + np.array(goal)) / 2
    dist = np.linalg.norm((np.array(x_start), np.array(goal)))
    slopez = (goal[2] - x_start[2]) / dist
    slopey = (goal[1] - x_start[1]) / dist
    slopex = (goal[0] - x_start[0]) / dist
    perp_mid = [midpoint[0] + cx * slopex, midpoint[1] + cy * slopey, midpoint[2] + cz * slopez]
    U = opti.variable(nu, N)  # control trajectory (throttle)
    T = 10 * ((X[0, -1] - goal[0]) ** 2 + (X[1, -1] - goal[1]) ** 2 + (X[2, -1] - goal[2]) ** 2) + 10 * (
                (X[0, N / 2] - perp_mid[0]) ** 2 + (X[1, N / 2] - perp_mid[1]) ** 2 + (X[2, N / 2] - perp_mid[2]) ** 2)
    for n in range(N):
        T += (1 * U[0, n] ** 2 + 1 * U[1, n] ** 2 + 1 * U[2, n] ** 2 + 1 * U[3, n] ** 2)

    # ---- objective          ---------
    opti.minimize(T)  # race in minimal time

    for k in range(N):  # loop over control intervals
        # Runge-Kutta 4 integration
        k1 = f(X[:, k], U[:, k])
        k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
        k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
        k4 = f(X[:, k] + dt * k3, U[:, k])
        x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

    # ---- path constraints -----------
    opti.subject_to(opti.bounded(-1, U, 1))  # control is limited

    # ---- boundary conditions --------
    opti.subject_to(pos[:, 0] == [0, 0, 0])  # start at position 0 ...
    opti.subject_to(speed[:, 0] == [0, 0, 0])  # ... from stand-still

    # ---- misc. constraints  ----------
    opti.subject_to(T >= 0)  # distance must be positive

    # ---- solve NLP              ------
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    opti.solver("ipopt", opts)  # set numerical backend
    sol = opti.solve()  # actual solve

    # ---- post-processing        ------
    solution = sol.value(pos)
    return solution

# goal = [5, 5, 5]
# x_start = [0, 0, 0]
# N = 20
# dt = 0.5
#
# # Bottom
# sol_bot = generate_traj(x_start, goal, N, dt, 0, 3, -3)
#
# # Top
# sol_top = generate_traj(x_start, goal, N, dt, 0, -3, 3)
#
# # Left
# sol_lef = generate_traj(x_start, goal, N, dt, -3, 3, 0)
#
# # Right
# sol_rig = generate_traj(x_start, goal, N, dt, 3, -3, 0)
#
# # Middle
# sol_mid = generate_traj(x_start, goal, N, dt, 0, 0, 0)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.plot(sol_bot[0, :], sol_bot[1, :], sol_bot[2, :], label="bottom")
# ax.plot(sol_top[0, :], sol_top[1, :], sol_top[2, :], label="top")
# ax.plot(sol_lef[0, :], sol_lef[1, :], sol_lef[2, :], label="left")
# ax.plot(sol_rig[0, :], sol_rig[1, :], sol_rig[2, :], label="right")
# ax.plot(sol_mid[0, :], sol_mid[1, :], sol_mid[2, :], label="bottom")
# ax.legend(loc="upper left")
# plt.show()
