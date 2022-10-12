# libs
import numpy as np
import numpy.linalg as la
import mmo

####################################################################################################
# config
####################################################################################################
N_SOL = 3
DIM = 2
BUDGET = 10000000

####################################################################################################
# solutions, objective function and domain
####################################################################################################
solutions = 0.1 + 0.8 * np.random.rand(N_SOL, DIM)

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
mmm = mmo.MultiModalMinimizer(f = f, domain = dom, budget = BUDGET, verbose = 1)
for m in mmm:
    print(m)

    print()
    for r in m.domain.regions:
        print(r)

    m.domain.plot(x = solutions)


sols = 1 + np.random.rand(N_SOL, DIM)

r = dom.pop_region()
r1, r2 = r.bisect(axis = 0, eta = 0.25)
dom.push_regions([r1, r2])
print(dom)

dom.plot(x = sols)

