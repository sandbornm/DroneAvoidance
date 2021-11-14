import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from generate_traj import generate_traj
from lemke_howson import LCP_lemke_howson


""" hyperparams """
T = 30 # game length
L = 5 # turn length
H = 10 # predictive horizon
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
def intersect(a, b, tol=1): # TODO adjust tol as needed
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

"""
takes the vector of 4D control inputs over the H steps of the computed trajectory
"""
def accPenalty(accs):
    print("in accpenalty")
    print("accs.shape")
    print(accs.shape)
    avAcc = np.average(accs, axis=1)
    print(avAcc.shape)
    print(avAcc.T.dot(avAcc))
    return avAcc.T.dot(avAcc)



""" get cost functions to be solved for each

Za: the state vector for A 
Ua: the control input for A
Same for Zb, Ub
"""
def getCostMats(Za, Ua, Zb, Ub):
    mats = []

    # for turn in range(nT):
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N): # TODO are these costs symmetric?
            #print(f"Za shape {Za[i, :, :].shape}")
            A[i, j] = distToGoal(Za[i, :, :]) + (1e6 * intersect(Za[i, :, :], Zb[j, :, :]))  +(.5 * accPenalty(Ua[i, :, :]))
            B[i, j] = distToGoal(Zb[i, :, :], player="b") + (1e6 * intersect(Za[j, :, :], Zb[i, :, :])) + (.5 * accPenalty(Ub[i, :, :]))
    mats.append([A, B])
    return np.array(mats)


"""" solve the LCPs given by A and B at each turn in the game using nashpy """
def solveLCPs(mats):
    print("in solve LCPS")
    print(type(mats))
    print(type(mats[0]))
    sols = []
    for x in list(mats):
        #print(x[0], x[1])
        #gme = nash.Game(x[1], x[0])
        #sols.append(lemke_howson_lex(x[0], x[1]))
        sols.append(LCP_lemke_howson(x[0], x[1]))
    return np.array(sols)


""" 
xcur:
goal: x,y,z goal for player
"""
def generateTrajectories(start, vstart, goal, cx, cy, cz):
    traj, acc = [], []
    for i in range(N):
        state, control = generate_traj(start, vstart, goal, H, dt, cx[i], cy[i], cz[i])
        traj.append(state)
        acc.append(control)
    return np.array(traj), np.array(acc)





""" visualize the results of the simulation """
def sim(startA, goalA, startB, goalB, cx, cy, cz):
    
    # initial start

    print("startA")
    print(startA)
    print("startB")
    print(startB)
    nextStartA = startA
    nextStartB = startB
    vStartA = [0,0,0]
    vStartB = [0,0,0]

    # store actual chosen trajectories
    trajAactual = np.zeros((dof, T))
    trajBactual = np.zeros((dof, T))

    for turn in range(nT):

        print(f"nextStartA {nextStartA}")
        print(f"nextStartB {nextStartB}")

        trajA, accA = generateTrajectories(nextStartA, vStartA, goalA, cx, cy, cz)
        trajB, accB = generateTrajectories(nextStartB, vStartB, goalB, cx, cy, cz)

        # print("traj and acc shapes")
        # print(f"trajA {trajA.shape} accA {accA.shape}")
        # only consider first 10 points
        costs = getCostMats(trajA, accA, trajB, accB)

        sols = solveLCPs(costs)

        #print(f"sols shape {sols.shape}")

        # choose action
        asol = np.argmax(sols[0,0])
        bsol = np.argmax(sols[0,1])
        #print(asol, bsol)

        # update actual trajectories
        trajAactual[:, turn*L:(turn+1)*L] = trajA[asol, :, 0:L]
        trajBactual[:, turn*L:(turn+1)*L] = trajB[bsol, :, 0:L]

        # update next start position
        nextStartA = trajA[asol, :3, L] # assign next start to xyz
        nextStartB = trajB[bsol, :3, L]

        # update next start velocity
        vStartA = trajA[asol, 4:7, L] # assign next velocity to xyz
        vStartB = trajB[bsol, 4:7, L]

        print("actual traj") # TODO add error in each direction
        print(f"A {trajAactual[:3, -1]}")
        print(f"B {trajBactual[:3, -1]}")

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #ax.plot(trajA[asol, 0, :], trajA[asol, 1, :], trajA[asol, 2, :], label="A")
        #ax.plot(trajB[bsol, 0, :], trajB[bsol, 1, :], trajB[bsol, 2, :], label="B")

        ax.plot(trajAactual[0, :], trajAactual[1, :], trajAactual[2, :], label="A")
        ax.plot(trajBactual[0, :], trajBactual[1, :], trajBactual[2, :], label="B")
        ax.legend(loc="upper left")
        plt.show()






    # TODO animation
    # TODO % error in each direction
    # TODO other metrics for evaluation?

    # fig, ax = plt.subplots()
    # # pos_a = ax.scatter(Zactual_a[0, 0], Zactual_a[1, 0], 10, 'b')
    # # pos_b = ax.scatter(Zactual_b[0,0], Zactual_b[1, 0], 10, 'r')

    # def animate(t):
    #     pass
    #     # pos_a.set_offsets([Zactual_a[0, t], Zactual_a[1, t]])
    #     # pos_b.set_offsets([Zactual_b[0, t], Zactual_b[1, t]])

    # ani = animation.FuncAnimation(fig,
    #                                 animate,
    #                                 save_count=T,  # total number of calls to animate
    #                                 interval=dt * 1000)  # interval = miliseconds between frames
    # ani.save("tag.mp4")




""" Driver code """
startA = [5, 5, 5] # blue
goalA = [0, 0, 0]
startB = [0, 0, 0] # orange
goalB = [5, 5, 5]
cx = [0, 0, -3, 3, 0] # TODO change these?
cy = [3, -3, 3, -3, 0]
cz = [-3, 3, 0, 0, 0]
sim(startA, goalA, startB, goalB, cx, cy, cz)

# generate trajectories

# print(trajA.shape, accA.shape)
# print(trajB.shape, accB.shape)

# get costs
# costs = getCostMats(trajA, trajB)
# print(f"costs[0] shape {costs[0].shape}")
# print(costs)

# # solve the LCPs
# sols = solveLCPs(costs)
# print(f"solution shape {sols.shape}")
# print(sols)
# print(sols.shape)
# print(type(sols))

# asol = np.argmax(sols[0,0])
# bsol = np.argmax(sols[0,1])
# print(asol, bsol)

# print("done")

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.plot(trajA[asol,0, :], trajA[asol,1,:], trajA[asol,2,:], label="A")
# ax.plot(trajB[bsol,0, :], trajB[bsol,1,:], trajB[bsol,2,:], label="B")
# ax.legend(loc="upper left")
# plt.show()




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