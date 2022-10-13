#!/usr/bin/env python

# std libs
import numpy as np
import numpy.linalg as la
from copy import deepcopy as copy
import mmo

###############################################################################
# classes
###############################################################################
class MultiModalMinimizer:
    def __init__(self, f = None, domain = None, verbose = 0, budget = np.inf, max_iter = 10**20, tol = 1e-8):
        assert(f is not None)
        assert(domain is not None)
        self.f = f
        self.domain = domain
        self.dim = domain.dim
        self.solutions_x = np.zeros((0, self.dim))
        self.solutions_y = np.zeros(0)
        self.budget = budget
        self.max_iter = max_iter
        self.verbose_1 = verbose >= 1
        self.verbose_2 = verbose >= 2
        self.verbose_3 = verbose >= 3
        self.n_fct_calls = 0
        self.n_local_solves = 0
        self.tol = tol

    def fct(self, x):
        self.n_fct_calls += 1
        return(self.f(x))

    def __iter__(self):
        self.iter = 0
        return(self)

    def __str__(self):
        s = ''
        if self.verbose_1:
            s += '## MultiModalMinimizer\n'
            s += f'iteration: {self.iter - 1}\n'
            s += f'n_local_solves: {self.n_local_solves}\n'
            s += f'n_fct_calls: {self.n_fct_calls}\n'
            s += f'n_solutions: {self.solutions_x.shape[0]}'
        return(s)

    def __next__(self):
        # search in region
        r = self.domain.pop_region()
        sigma = 0.1 * np.sqrt((r.ur[0] - r.ll[0]) * (r.ur[1] - r.ll[1]))
        cma = mmo.Cma(f = self.fct, x0 = 0.5 * (r.ll + r.ur) , sigma = sigma)

        # update domain
        r1, r2 = r.bisect()
        e1, i1, p1 = r1 < cma.x
        e2, i2, p2 = r2 < cma.x

        # handle cases
        solution_new = True
        if p1 + p2 == 0:
            # cma.x was not put in
            if e1 + e2 == 0:
                # weird case, both not empy"
                #print("## r1, r2 not empty")
                assert(0)
            if i1 + i2 == 0:
                # converged outside
                #print("## cma.x outside of r")
                r1.mod(not_converged = True)
                r2.mod(not_converged = True)
                found = False
                for r in self.domain.regions:
                    if r.is_in(cma.x):
                        found = True
                        break
                if found:
                    #print("#### found region for cma.x")
                    # found a region, where solution is in
                    if r.p is None:
                        #print("###### region is empty")
                        # region is empty
                        r.mod(p = cma.x)
                    else:
                        self.domain.pop_region(r)
                        # region is not empty
                        #print("###### region is not empty")
                        if la.norm(cma.x - r.p) < self.tol:
                            # solution are the same
                            solution_new = False
                            #print("######## solution is the same")
                            r3, r4 = r.bisect()
                            r3.mod(solutions_same = True)
                            r4.mod(solutions_same = True)
                            self.domain.push_regions([r3, r4])
                        else:
                            #print("######## solution is not the same, finding split parameters")
                            # solution are not same, need to split
                            mp = 0.5 * (r.p + cma.x)
                            axis, eta = r.split_parameters(mp)
                            r3, r4 = r.bisect(axis = axis, eta = eta)
                            r3 < cma.x
                            r4 < cma.x
                            self.domain.push_regions([r3, r4])

                else:
                    #print("#### did not find region for cma.x")
                    # no region found
                    assert(1)

        self.domain.push_regions([r1, r2])

        
        # save solution
        self.n_local_solves += 1
        if solution_new:
            self.solutions_x = np.vstack((self.solutions_x, cma.x))
            self.solutions_y = np.hstack((self.solutions_y, cma.y))

        # stop
        if self.n_fct_calls >= self.budget or self.iter >= self.max_iter:
            raise StopIteration

        # admin
        self.iter += 1
        return(copy(self))







