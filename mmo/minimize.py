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
            s += f'n_fct_calls: {self.n_fct_calls}\n'
        return(s)

    def __next__(self):
        # search in region
        rs = self.domain.get_top_region()
        cma = mmo.Cma(f = self.fct, region = rs)
        self.n_local_solves += 1
        rr = self.domain.get_region_with_point(cma.x)

        if rs == rr:
            # solution found in r0
            r1, r2 = rs.bisect(p = cma.x)
            self.domain.replace(regions_in = [r1, r2], regions_out = [rs])

        else:
            r1, r2 = rs.bisect(p = None)
            r3, r4 = rr.bisect(p = cma.x)
            self.domain.replace(regions_in = [r1, r2, r3, r4], regions_out = [rs, rr])

        # stop
        if self.n_fct_calls >= self.budget or self.iter >= self.max_iter:
            raise StopIteration

        # admin
        self.iter += 1
        return(copy(self))







