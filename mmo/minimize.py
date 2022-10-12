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
    def __init__(self, f = None, domain = None, verbose = 0, budget = np.inf, max_iter = 10**20):
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
            s += f'n_fct_calls: {self.n_fct_calls}'
        return(s)

    def __next__(self):
        # search in region
        r = self.domain.pop_region()
        cma = mmo.Cma(f = self.fct, x0 = 0.5 * (r.ll + r.ur) , sigma = 0.1 * np.sqrt((r.ur[0] + r.ll[0]) * (r.ur[1] + r.ll[1])))

        # update domain
        r1, r2 = r.bisect()
        e1, i1, p1 = r1 < cma.x
        e2, i2, p2 = r2 < cma.x

        # handle cases
        if p1 + p2 == 0:
            # cma.x was not put in
            if e1 + e2 == 0:
                print("# weird case, both not empy")
                assert(0)
            if i1 + i2 == 0:
                print("# converged outside")
                assert(0)

        # put regions in
        self.domain.push_regions([r1, r2])
        
        # save solution
        self.n_local_solves += 1
        self.solutions_x = np.vstack((self.solutions_x, cma.x))
        self.solutions_y = np.hstack((self.solutions_y, cma.y))

        # admin
        self.iter += 1
        return(copy(self))







