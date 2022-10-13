# libs
import numpy as np
import numpy.linalg as la
import mmo
from scipy.spatial import distance_matrix

####################################################################################################
# config
####################################################################################################
N_SOL = 10
DIM = 2
BUDGET = 10000000

####################################################################################################
# solutions, objective function and domain
####################################################################################################
solutions = 0.1 + np.array([0.01, 0.8]).reshape(1, DIM) * np.random.rand(N_SOL, DIM)

def solutions_found(x):
    dm = distance_matrix(x, solutions)
    dm = np.min(dm, axis = 0)
    n = np.sum(dm < 1e-10)
    return(n)

def f(x):
    r = la.norm(solutions - x.reshape(1, -1), axis = 1)
    return(np.min(r))

dom = mmo.Domain(ll = [0]*DIM, ur = [1]*DIM)

####################################################################################################
# run
####################################################################################################
mmm = mmo.MultiModalMinimizer(f = f, domain = dom, budget = BUDGET, verbose = 1, max_iter = 100)
for k, m in enumerate(mmm):
    print(m)
    print()
    n = solutions_found(m.solutions_x)
    print(f'solutions found: {n}')
    print()
    if n == N_SOL:
        break

m.domain.plot(x = solutions)



