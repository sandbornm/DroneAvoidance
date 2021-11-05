from matplotlib import cm
import numpy as np
from casadi import *  # "python3 -mpip install casadi" should do the trick
import matplotlib.pyplot as plt

from path_corrected import path_solver

import matplotlib.patches as ptc


# Define a general optimization problem with non-quadratic cost or nonlinear constraints

T = 50
n = 4
m = 2
dt = 0.1

xinit = np.zeros((4,1))
rad = 0.25
obs = [2,1.]
nonlinear_dyn = False
if nonlinear_dyn:
    goal = [-4,1]
else:
    goal = [3, 1]

""" change this to True for other behavior """
use_other = True
if use_other:
    # add another circle
    obs2 = [1,.8]
    rad2 = 0.3

# Simple car dynamics
def f(xt,ut):
    print(xt)
    print(f"type(xt) {type(xt)}, xt.shape {xt.shape}")
    print(ut)
    print(f"type(ut) {type(ut)}, ut.shape {ut.shape}")
    y = []
    if nonlinear_dyn:
        y.append(xt[0]+dt*xt[2]*cos(xt[3]))
        y.append(xt[1]+dt*xt[2]*sin(xt[3]))
        y.append(xt[2]+dt*ut[0])
        y.append(xt[3]+dt*ut[1])
    else:
        y.append(xt[0]+dt*xt[2])
        y.append(xt[1]+dt*xt[3])
        y.append(xt[2]+dt*ut[0])
        y.append(xt[3]+dt*ut[1])
    return y

dynamics = []

U = []
X = []
x0 = MX.sym('x0',n)
X.append(x0)
dynamics.append(x0-xinit)
for t in range(T):
    ut = MX.sym('u%s'%t,m)
    xtt = MX.sym('x%s'%(t+1),n)
    pred = vcat(f(X[-1],ut))
    dynamics.append(xtt-pred)
    U.append(ut)
    X.append(xtt)

if use_other: # other cost
    cdeg = 2
    cmul = 16
    cost = cmul*((X[-1][0]-goal[0])**cdeg + (X[-1][1]-goal[1])**cdeg) 
    for t in range(T):
        cost += (1* U[t][0]**2 + 1*U[t][1]**2)
else: # default cost
    cost = 10*((X[-1][0]-goal[0])**2 + (X[-1][1]-goal[1])**2)
    for t in range(T):
        cost += (1*U[t][0]**2 + 1*U[t][1]**2)


constraints = []
for t in range(T):
    state = X[t+1]
    if nonlinear_dyn:
        constraints.append(2-state[0])
        constraints.append(state[0]+2)
        constraints.append(2-state[1])
        constraints.append(state[1]+2)
    else:
        const = (state[0]-obs[0])**2 + (state[1]-obs[1])**2 - rad*rad # square distance from obs should be rad*rad
        constraints.append(const)
        if use_other: # constraint to avoid obs2
            const2 = (state[0]-obs2[0])**2 + (state[1]-obs2[1])**2 - rad2*rad2
            constraints.append(const2)

# min_{X,U} cost
#  s.t. dynamics = 0
#       constraints >= 0

# Form KKT conditions

# X shape (51, 4)
# U shape (50, 2)
all_dyn = vcat(dynamics) # shape (204, 1)
all_ineq = vcat(constraints) # shape  (50, 1)
all_primal_vars = vcat(X+U) # shape (304, 1)

dyn_mults = MX.sym('dyn_mults', all_dyn.shape[0]) # shape (204, 1)
ineq_mults = MX.sym('ineq_mults', all_ineq.shape[0]) # shape (50, 1)

lag = cost - dot(all_dyn,dyn_mults) - dot(all_ineq,ineq_mults) # scalar

dlag = jacobian(lag, all_primal_vars) # shape (1, 304)
kkt_expr = vcat([dlag.T, all_dyn, all_ineq]) # shape (558, 1)
all_vars = vcat([all_primal_vars,dyn_mults,ineq_mults]) # shape (558, 1)
jac_kkt = jacobian(kkt_expr, all_vars) # shape (558, 558)

# functions to evaluate f and df derived from f(xt, ut) above
eval_kkt = Function('kkt',[all_vars],[kkt_expr])
eval_kkt_jac = Function('kkt',[all_vars],[jac_kkt])

n = all_vars.shape[0] 
nprimal = (T+1)*4 + T*2
ndyn = (T+1)*4
nineq = all_ineq.shape[0]

def feval(y):
    return np.array(eval_kkt(y))

def dfeval(y):
    return np.array(eval_kkt_jac(y))

l = np.vstack((-np.inf*np.ones((nprimal+ndyn,1)),np.zeros((nineq,1))))

if use_other: # other upper bound
    u = 500 * np.ones((n,1))
else:
    u = np.inf*np.ones((n,1))


u_start = np.zeros((T*2,1))
if nonlinear_dyn:
    u0 = np.array([0.1,-0.01]).reshape(2,1)
else:
    u0 = np.array([0,1]).reshape(2,1)

x_start = np.zeros(((T+1)*4,1))
x_start[0:4] = xinit
for t in range(T):
    u_start[t*2:(t+1)*2] = u0
    x_start[(t+1)*4:(t+2)*4] = f(x_start[(t)*4:(t+1)*4], u_start[(t)*2:(t+1)*2])

x0 = np.vstack((x_start,u_start,np.zeros(((ndyn+nineq,1)))))

[z,w,v,success] = path_solver(feval, dfeval, l, u, x0=x0, sigma=0.1, max_iters=100, tol=1e-4,linesearch=nonlinear_dyn)
print(f"success: {success}")
traj = z[0:(T+1)*4]
px = traj[0::4]
py = traj[1::4]

plt.plot(goal[0],goal[1],'yo') # goal
plt.annotate("goal", goal)
if nonlinear_dyn:
    plt.plot([-2,-2,2,2,-2],[-2,2,2,-2,-2])
else:
    R = np.linspace(0,6.3,100)
    plt.plot(rad*cos(R)+obs[0],rad*sin(R)+obs[1]) # obs
    
    if use_other: # add other obstacle
        plt.annotate("obs2", obs2)
        plt.plot(rad2*cos(R)+obs2[0],rad2*sin(R)+obs2[1]) # obs2
        plt.title(f"obs2 center={obs2}, radius={rad2}")
        """ change these titles based on the values of interest """
        #plt.title(f"max degree of cost terms: {cdeg}")
        #plt.title(f"upper bound u={u[0]}")
        #plt.title(f"cost multiplier {cmul}")
plt.figure
plt.plot(px,py)
plt.axis([-4,4,-4,4])
plt.axis('square')
plt.show()
