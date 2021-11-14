import nashpy as nash
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from generate_traj import generate_traj

from HW2 import LCP_lemke_howson


""" hyperparams """
T = 3 # game length
L = 3 # turn length
H = 20 # predictive horizon
nT = int(T/L) # number of turns in game
N = 5 # number of candidate trajectories
dof = 8
dt = .5

""" dummy trajectories """
# assume dof contains (x,y,z) spatial information in the first 3 elements
# trajA = np.random.randint(low=10, high=20, size=(nT, N, dof, H))
# trajB = np.random.randint(low=10, high=20, size=(nT, N, dof, H))


""" check if a single pair of trajectories a and b intersect at any point 
    to be used to weight against collision """
def intersect(a, b, tol=.3): # TODO adjust tol as needed
    assert a.shape == (dof, H+1) and b.shape == (dof, H+1)
    dists = []
    for h in range(H):
        axyz = a[:3, h]
        bxyz = b[:3, h]
        dists.append(np.linalg.norm(axyz - bxyz))
    dists = np.array(dists)
    # print(np.array([int(d < tol) for d in dists]))
    # print(int(np.any(dists < tol)))
    return int(np.any(dists < tol))


""" get the minimum distance to goal (L2 norm in 3-space) over the next H trajectories """
def distToGoal(traj, player="a"):
    #traj = traj.T # will be (8, 21)
    #print(f"traj shape {traj.shape}")
    goal = goalA if player == "a" else goalB
    dist = np.inf
    dist = min(dist, np.linalg.norm(traj[:3, H] - goal))
    return dist


""" get cost functions to be solved for each"""
def getCostMats(Za, Zb):
    mats = []
    # for turn in range(nT):
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N): # TODO are these costs symmetric?
            #print(f"Za shape {Za[i, :, :].shape}")
            A[i, j] = distToGoal(Za[i, :, :]) + (1e6 * intersect(Za[i, :, :], Zb[j, :, :]))
            B[i, j] = distToGoal(Zb[i, :, :], player="b") + (1e6 * intersect(Za[j, :, :], Zb[i, :, :]))
    mats.append([A, B])
    return np.array(mats)


"""" solve the LCPs given by A and B at each turn in the game using nashpy """
def solveLCPs(mats):
    sols = []
    for [A,B] in list(mats):
        gme = nash.Game(A, B) # todo how to deal with this
        sols.append(gme.lemke_howson(initial_dropped_label=0))
        #sols.append(LCP_lemke_howson(A, B))
    return np.array(sols)


""" 
xcur:
goal: x,y,z goal for player
"""
def generateTrajectories(start, goal, cx, cy, cz):
    traj = []
    for _ in range(nT):
        for i in range(N):
            t = generate_traj(start, goal, H, dt, cx[i], cy[i], cz[i])
            traj.append(t)
    return np.array(traj)


""" visualize the results of the simulation """
def sim():
    fig, ax = plt.subplots()
    # pos_a = ax.scatter(Zactual_a[0, 0], Zactual_a[1, 0], 10, 'b')
    # pos_b = ax.scatter(Zactual_b[0,0], Zactual_b[1, 0], 10, 'r')

    def animate(t):
        pass
        # pos_a.set_offsets([Zactual_a[0, t], Zactual_a[1, t]])
        # pos_b.set_offsets([Zactual_b[0, t], Zactual_b[1, t]])

    ani = animation.FuncAnimation(fig,
                                    animate,
                                    save_count=T,  # total number of calls to animate
                                    interval=dt * 1000)  # interval = miliseconds between frames
    ani.save("tag.mp4")



""" Driver code """
startA = [1, 1, 1]
goalA = [0, 0, 0]
startB = [0, 0, 0]
goalB = [1, 1, 1]
cx = [0, 0, -3, 3, 0]
cy = [3, -3, 3, -3, 0]
cz = [-3, 3, 0, 0, 0]

# generate trajectories
trajA = generateTrajectories(startA, goalA, cx, cy, cz)
trajB = generateTrajectories(startB, goalB, cx, cy, cz)
print(trajA.shape)
print(trajB.shape)

# get costs
costs = getCostMats(trajA, trajB)
print(costs)

# solve the LCPs
sols = solveLCPs(costs)
print(f"solution shape {sols.shape}")
print(sols)

asol = np.argmax(sols[0,0])
bsol = np.argmax(sols[0,1])
print(asol, bsol)

print("done")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(trajA[asol,0, :], trajA[asol,1,:], trajA[asol,2,:], label="A")
ax.plot(trajB[bsol,0, :], trajB[bsol,1,:], trajB[bsol,2,:], label="B")
ax.legend(loc="upper left")
plt.show()




# mats = getCostMats(trajA, trajB)
# sols = solveLCPs(mats)

# Test RPS bimatrix example
# A = np.array([[2, 3, 1], [1, 2, 3], [3, 1, 2]])
# B = -copy.deepcopy(A)

# gme = nash.Game(A, B)
# x, y = gme.lemke_howson(initial_dropped_label=0)
# print(x)
# print(y)
# assert np.linalg.norm(x - y) <= 1e-6
# assert np.linalg.norm(x - 1 / 3.0 * np.array([1, 1, 1])) <= 1e-6