# libs
import numpy as np
import numpy.linalg as la
import mmo
from scipy.spatial import distance_matrix

####################################################################################################
# config
####################################################################################################
N_SOL = 100
DIM = 2 
BUDGET = 1000000000

####################################################################################################
# solutions, objective function and domain
####################################################################################################
alpha = 2* 3.141562 * np.random.rand(N_SOL)
solutions = 0.5 + 0.4* np.vstack((np.sin(alpha), np.cos(alpha))).T
N_SOL = solutions.shape[0]

def solutions_found(x):
    dm = distance_matrix(x, solutions)
    dm = np.min(dm, axis = 0)
    n = np.sum(dm < 1e-8)
    return(n)

def f(x):
    r = la.norm(solutions - x.reshape(1, -1), axis = 1)
    return(np.min(r) ** 0.5)

dom = mmo.Domain(ll = [0]*DIM, ur = [1]*DIM)

####################################################################################################
# run
####################################################################################################
mmm = mmo.MultiModalMinimizer(f = f, domain = dom, budget = BUDGET, verbose = 1, max_iter = 100000)
for m in mmm:
    print(m)
    print()
    n = solutions_found(m.domain.solutions())
    print(f'solutions found: {n}')
    print()
    if n == N_SOL:
        break

m.domain.plot(x = solutions)



