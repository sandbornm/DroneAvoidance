import nashpy as nash
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


""" hyperparams """
T = 30 # number of timesteps
L = 3 # turn length
H = 20 # predictive horizon
nT = int(T/L) # number of turns in game
N = 5 # number of candidate trajectories
goalA = np.zeros((N, 3)) # goal position [x, y] for each trajectory, to do change to 3
goalB = np.ones((N, 3))
dof = 12
dt = .5

""" dummy trajectories """
# assume dof contains (x,y,z) spatial information in the first 3 elements
trajA = np.random.randint(low=10, high=20, size=(nT, N, dof, H))
trajB = np.random.randint(low=10, high=20, size=(nT, N, dof, H))


""" check if a single pair of trajectories a and b intersect at any point 
    to be used to weight against collision """
def intersect(a, b, tol=.3): # TODO adjust tol as needed
    assert a.shape == (dof, H) and b.shape == (dof, H)
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
    goal = goalA if player == "a" else goalB
    dist = np.inf
    dist = min(dist, np.linalg.norm(traj[:3, H-1] - goal))
    return dist


""" get cost functions to be solved for each"""
def getCostMats(Za, Zb):
    mats = []
    for turn in range(nT):
        A = np.zeros((N, N))
        B = np.zeros((N, N))
        for i in range(N):
            for j in range(N): # TODO are these costs symmetric?
                A[i, j] = distToGoal(Za[turn, i, :, :]) + (10e6 * intersect(Za[turn, i, :, :], Zb[turn, j, :, :]))
                B[i, j] = distToGoal(Zb[turn, i, :, :], player="b") + (10e6 * intersect(Za[turn, j, :, :], Zb[turn, i, :, :]))
        mats.append([A, B])
    return mats


"""" solve the LCPs given by A and B at each turn in the game using nashpy """
def solveLCPs(mats):
    sols = []
    for [A,B] in mats[:1]:
        gme = nash.Game(A, B)
        sols.append(gme.lemke_howson(initial_dropped_label=0))
    return sols



""" dummy function for generate trajectory """
def gT(xcur, goal):
    print(xcur, goal)
    return 1

""" 
xcur: current state vector 12 dof
goal: goal for player in 3-space
"""
def GenerateTrajectories(xcur, goal):
    traj = []
    for _ in range(nT):
        for _ in range(N):
            t = gT(xcur, goal, H, dt)
            traj.append(t)
    return traj


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

mats = getCostMats(trajA, trajB)
sols = solveLCPs(mats)

# Test RPS bimatrix example
# A = np.array([[2, 3, 1], [1, 2, 3], [3, 1, 2]])
# B = -copy.deepcopy(A)

# gme = nash.Game(A, B)
# x, y = gme.lemke_howson(initial_dropped_label=0)
# print(x)
# print(y)
# assert np.linalg.norm(x - y) <= 1e-6
# assert np.linalg.norm(x - 1 / 3.0 * np.array([1, 1, 1])) <= 1e-6