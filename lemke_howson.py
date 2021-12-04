# HW1_sol.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import copy


def LCP_lemke_howson(A, B, max_iters=50):
    # Function for computing solutions to the Bimatrix game defined by cost matrices A > 0, B > 0
    # Returns a tuple (x,y), where x and y are the mixed equilibrium strategies for player 1 and player 2, respectively.

    # This function is an implementation of the Lemke-Howson method, as described in Section 2.5 of
    # "Linear Complementarity, Linear and Nonlinear Programming", by Katta G. Murty
    n = A.shape[0]
    m = A.shape[1]

    # Ensure A and B are strictly positive
    if (np.any(A <= 0)): # add separately to handle ufunc error
        A = A + abs(np.min(A)) + float(1.0) # any positive constant would do
    if (np.any(B <= 0)):
        B = B + abs(np.min(B)) + float(1.0) # any positive constant would do
    
    M = np.vstack((np.hstack((np.zeros((n,n)), A)), np.hstack((B.T, np.zeros((m,m))))))
    T = np.hstack((np.eye(n+m), -M, -np.ones((n+m, 1))))
    basis = np.arange(n+m)

    enter_idx = n + m # start with last column of T corresponding to q
    t = np.argmin(-T[n:n+m, enter_idx]) + n
    basis[t] = enter_idx # update basis

    pivot = T[t, :] / T[t, enter_idx] # similar to above, assign entering var st exiting var is 0
    T -= np.outer(T[:, enter_idx], pivot) # row reduce
    T[t, :] = pivot

    enter_idx = t + n + m # next entering variable, complement of previous
    t = np.argmin(-T[0:n, enter_idx])
    basis[t] = enter_idx

    pivot = T[t, :] / T[t, enter_idx]
    T -= np.outer(T[:, enter_idx], pivot)
    T[t, :] = pivot

    enter_idx = t+n+m
    xi1 = n+m
    u1 = 0

    T_start = copy.deepcopy(T)

    it, ret = 0, 1 # assume no ray termination
    while (xi1 in basis and u1 in basis and it < max_iters):
        it +=1
        d = T[:, enter_idx]
        wrong_dir = d <= 0
        ratios = np.zeros(n+m)

        for i in range(n+m):
            if d[i] > 0: # get ratios
                ratios[i] = T[i, -1] / d[i]
        ratios[wrong_dir] = np.inf

        t = np.argmin(ratios)
        minRat = ratios[t]
        doLex = True

        if np.sum(np.min(ratios) == minRat) > 1: # check for duplicates 
            if ratios[np.where(basis == u1)] == minRat:
                t = np.where(basis == u1)[0][0]
                doLex = False
            elif ratios[np.where(basis == xi1)] == minRat:
                t = np.where(basis == xi1)[0][0]
                doLex = False
            else:
                B_inv = np.linalg.inv(T_start[:, basis])

            candidate_idx = [] # candidate indices
            for i in range(n):
                if d[i] > 0 and ratios[i] == np.min(ratios): # ignore non-positive values
                    candidate_idx.append(i)

            k = 0
            while doLex and np.sum(ratios == np.min(ratios)) > 1 and k < n+m:
                wrong_dir = d[candidate_idx] <= 0
                ratios = B_inv[candidate_idx, i] / d[candidate_idx]
                ratios[wrong_dir] = np.inf
                t = candidate_idx[np.argmin(ratios)] # store current min
                throwaway_idx = [] # remove idxs that dont correspond to min
                for i in range(len(candidate_idx)):
                    if ratios[i] != np.min(ratios):
                        throwaway_idx.append(candidate_idx[i])
                for j in range(len(throwaway_idx)):
                    candidate_idx.remove(throwaway_idx[j])
                k += 1

        if ~ np.all(wrong_dir):
                # This block of code assigns entering var necessary value so that exiting var is 0
                pivot = T[t, :] / T[t, enter_idx]
                T -= np.outer(T[:, enter_idx], pivot)
                T[t, :] = pivot
                #
                exiting_ind = basis[t]
                basis[t] = enter_idx
                if exiting_ind >= n+m:
                    enter_idx = exiting_ind - n - m
                else:
                    enter_idx = exiting_ind + n + m
        else:
            ret = 0
            print("ray termination")
            break

    if ret:
        vars = np.zeros((m+n, 1))
        align = basis % (n+m) >= 0
        vars[basis[align] % (n+m), 0] = T[align, -1]
        xi = vars[0:n]
        eta = vars[n:m+n]
        x = xi / np.sum(xi)
        y = eta / np.sum(eta)
        return (x, y)
    else:
        x = np.zeros((n, 1))
        y = np.zeros((m, 1))
        return (x, y) 
