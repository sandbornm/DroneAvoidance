import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D

from generate_traj import generate_traj
from lemke_howson import LCP_lemke_howson
import os
from tqdm import tqdm
import time


""" hyperparams - can be adjusted for a parameter study"""
T = 100 # game length (25, 50, 100, 200)
L = 5 # turn length
H = 10 # predictive horizon
nT = int(T/L) # number of turns in game (5, 10, 20, 40)
N = 5 # number of candidate trajectories
dof = 8 # dof in dynamics model
dt = .1 # timestep size
tol = 1 # tolerance for checking if trajectories intersect




""" check if a single pair of trajectories a and b intersect at any point 
    to be used to weight against collision 
    
    a, b: np.array representing a trajectory of shape (dof, H+1) for a single turn A and B 

    returns: a single integer indicating whether the passed trajectories intersect within
            tolerance at any point in the trajectory
"""
def intersect(a, b):
    assert a.shape == (dof, H+1) and b.shape == (dof, H+1)
    dists = []
    for h in range(H):
        axyz = a[:3, h]
        bxyz = b[:3, h]
        dists.append(np.linalg.norm(axyz - bxyz))
    dists = np.array(dists)
    # print(np.array([int(d < tol) for d in dists]))
    #print(f"intersect return {int(np.any(dists < tol))}")
    return int(np.any(dists < tol))


""" get the minimum distance to goal (L2 norm in 3-space) over the next H trajectories 

    traj: np.array representing the trajectory of shape (dof, H+1) for the indicated player
    player: a string representing the player of interest
"""
def distToGoal(traj, player="a"):
    #print(f"traj shape {traj.shape}")
    goal = goalA if player == "a" else goalB
    dist = np.inf
    dist = min(dist, np.linalg.norm(traj[:3, H] - goal))
    return dist

"""
takes the vector of 4D control inputs over the H steps of the computed trajectory
"""
def accPenalty(accs):
    # print("in accpenalty")
    # print("accs.shape")
    # print(accs.shape)
    avAcc = np.average(accs, axis=1)
    # print(avAcc.shape)
    # print(avAcc.T.dot(avAcc))
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
    # print("in solve LCPS")
    # print(type(mats))
    # print(type(mats[0]))
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


"""
Compute the error at a single turn for the start position and goal for a single player

    start: the start position for the player
    goal: the goal position for the player

    returns: a list of percent error in the x, y, z directions

"""
def getErrorAtTurn(nextStart, goal):
    
    x, y, z = nextStart
    xg, yg, zg = goal

    xerr = round((abs(x-xg)/xg)*100, 2)
    yerr = round((abs(y-yg)/yg)*100, 2)
    zerr = round((abs(z-zg)/zg)*100, 2)

    err = [xerr, yerr, zerr]

    return err

"""
Run the simulation 

startA, B: the point in 3-space (x, y, z) where A/B starts from 
goalA, B: the point in 3-space (x, y, z) where A/B wants to go to

cx, cy, cz: scalar distance in each direction from the midpoint of the start to goal from
            the current trajectory

returns: a tuple of (tA, eA) - the actual trajectories for A and B and the percent error at each turn
         in each direction for A and B's distance to their respective goals
"""
def sim(startA, goalA, startB, goalB, cx, cy, cz):
    
    # print("startA")
    # print(startA)
    # print("startB")
    # print(startB)
    nextStartA = startA
    nextStartB = startB
    vStartA = [0,0,0]
    vStartB = [0,0,0]

    # store actual chosen trajectories
    trajAactual = np.zeros((dof, T))
    trajBactual = np.zeros((dof, T))
    
    # update initial error
    errorA = []
    errorB = []

    for turn in tqdm(range(nT)):

        # print(f"nextStartA {nextStartA}")
        # print(f"nextStartB {nextStartB}")

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

        #print(f"trajAactual {trajAactual.shape}")
        #print(f"trajBactual {trajAactual.shape}")

        # update next start position
        nextStartA = trajA[asol, :3, L] # assign next start to xyz
        nextStartB = trajB[bsol, :3, L]

        # update next start velocity
        vStartA = trajA[asol, 4:7, L] # assign next velocity to xyz
        vStartB = trajB[bsol, 4:7, L]

        #print(f"turn number {turn+1}/{nT}")

        # update error
        errorA.append(getErrorAtTurn(nextStartA, goalA))
        errorB.append(getErrorAtTurn(nextStartB, goalB))
        
    return (trajAactual, errorA), (trajBactual, errorB)


"""
Callback for animating the simulated game, 

t: the current iteration to update the figure from [0, T-1]
trajAactual: the actual path taken by vehicle A
trajBactual: the actual path taken by vehicle B
"""
def animate(t, fig, trajAactual, trajBactual):

    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"A and B trajectories with dt={dt}, nT={nT}, L={L}, H={H}, tol={tol}")
    l1 = plt.plot(trajAactual[0, 0], trajAactual[1, 0], trajAactual[2, 0], label="A")[0]
    l2 = plt.plot(trajBactual[0, 0], trajBactual[1, 0], trajBactual[2, 0], label="B")[0]
    #print(f"trajAactual shape {trajAactual.shape}")

    l1.set_data(trajAactual[0, :t], trajAactual[1, :t])
    l1.set_3d_properties(trajAactual[2, :t])
    l2.set_data(trajBactual[0, :t], trajBactual[1, :t])
    l2.set_3d_properties(trajBactual[2, :t])
    ax.legend(loc="lower left")

"""
save the animation as mp4 with title containing number of turns
and timestamp

ani: animation object representing simulated game
"""
def save(ani):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fname = f"results/avoid_{nT}_{timestr}.mp4"
    ani.save(fname)
    print(f"successfully saved {fname}")


"""
Plot the error for a player over all turns

    err: a list of (%x, %y, %z) tuples for % error loss at each turn for A/B

    return: plots for percent error at each turn of a player A/B
"""
def plotError(errA, errB):

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Percent error for A and B')
    ax1.plot(range(1, nT+1), [x[0] for x in errA], label="x error A", color='r')
    ax1.plot(range(1, nT+1), [x[1] for x in errA], label="y error", color='b')
    ax1.plot(range(1, nT+1), [x[2] for x in errA], label="z error", color='y')
    ax2.plot(range(1, nT+1), [x[0] for x in errB], label="x error B")
    ax2.plot(range(1, nT+1), [x[1] for x in errB], label="y error")
    ax2.plot(range(1, nT+1), [x[2] for x in errB], label="z error")
    ax1.legend()
    ax2.legend()
    plt.xlabel("turn number")
    plt.ylabel("percent error")
    #plt.show()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fname = f"results/error_{nT}_{timestr}.png"
    plt.savefig(fname)
    print(f"successfully saved {fname}")

""" Driver code """
startA = [6, 6, 6] # blue
goalA = [1, 1, 1]
startB = [1, 1, 1] # orange
goalB = [6, 6, 6]

# TODO change these?
cx = [0, 0, -3, 3, 0] 
cy = [3, -3, 3, -3, 0]
cz = [-3, 3, 0, 0, 0]

# do the simulation
(trajA, errA), (trajB, errB) = sim(startA, goalA, startB, goalB, cx, cy, cz)
print(f"len errA {len(errA), len(errA[0])}")

plotError(errA, errB)
# animation based on real trajectories
fig = plt.figure() 
ani = animation.FuncAnimation(fig,
                              animate,
                              save_count=T,
                              fargs = (fig, trajA, trajB),  # total number of calls to animate
                              interval=dt * 500)  # interval = miliseconds between frames
save(ani)

print("done")
