# libs
import numpy as np
import numpy.linalg as la
from copy import deepcopy as copy
import matplotlib.pyplot as plt

# fcts

# classes
class Region:
    def __init__(self, ll = None, ur = None, p = None, no_convergence = False):
        assert(ll is not None)
        assert(ur is not None)
        self.ll = ll
        self.ur = ur
        assert(self.ll.shape[0] == self.ur.shape[0])
        self.dim = self.ll.shape[0]
        self.p = p
        self.n_not_converged = 0

    def mod(self, p = None, not_converged = False):
        if p is not None:
            assert(self.p is None)
            self.p = p
        if not_converged:
            self.n_not_converged += 1

    def score(self):
        s = 1
        for k in range(self.ll.shape[0]):
            s *= self.ur[k] - self.ll[k]
        if self.p is not None:
            s *= 0.5
        s *= 0.5 ** self.n_not_converged
        return(s)

    def bisect(self, axis = None, eta = 0.5):
        if axis is None:
            l = -1
            for k in range(self.dim):
                if self.ur[k] - self.ll[k] > l:
                    l = self.ur[k] - self.ll[k]
                    axis = k
        r1 = copy(self)
        r1.p = None
        r2 = copy(self)
        r2.p = None
        r1.ur[axis] = self.ll[axis] + eta * (self.ur[axis] - self.ll[axis])
        r2.ll[axis] = r1.ur[axis]
        if self.p is not None:
            e1, i1, p1 = r1 < self.p
            e2, i2, p2 = r2 < self.p
            if p1 + p2 == 0:
                print("ERROR")
                print(f'e1 = {e1}, i1 = {i1}, p1 = {p1}')
                print(f'e2 = {e2}, i2 = {i2}, p2 = {p2}')
                exit()
        return(r1, r2)

    def is_in(self, x):
        is_in = True
        for k in range(self.dim):
            if x[k] < self.ll[k]: is_in = False
            if x[k] > self.ur[k]: is_in = False
        return(is_in)

    def __lt__(self, x):
        empty = 1 * (self.p is None)
        isin = 1 * self.is_in(x)
        putin = 0
        if empty + isin == 2:
            self.p = x
            putin = 1
        return(empty, isin, putin)

    def points(self):
        assert(self.dim == 2)
        points = np.array([ self.ll[0], self.ll[1], self.ur[0], self.ll[1], self.ur[0], self.ur[1], self.ll[0], self.ur[1], self.ll[0], self.ll[1] ]).reshape(5,2)
        return(points)

    def __str__(self):
        s = '# region\n'
        s += f'dim: {self.dim}\n'
        s += f'll: {self.ll}\n'
        s += f'ur: {self.ur}\n'
        s += f'p: {self.p}\n'
        s += f'score: {self.score()}\n'
        return(s)
    
class Domain:
    def __init__(self, ll = None, ur = None):
        assert(ll is not None)
        assert(ur is not None)
        self.ll = np.array(ll).astype(float)
        self.ur = np.array(ur).astype(float)
        assert(self.ll.shape[0] == self.ur.shape[0])
        self.dim = self.ll.shape[0]
        self.regions = [Region(ll = self.ll, ur = self.ur)]

    def pop_region(self):
        assert(len(self.regions) > 0)
        score_max = -np.inf
        for region in self.regions:
            score = region.score()
            if score > score_max:
                score_max = score
                region_max = region
        self.regions.remove(region_max)
        return(region_max)

    def push_regions(self, region):
        if isinstance(region, list):
            for r in region:
                self.push_regions(r)
        else:
            self.regions += [region]

    def plot(self, x = None):
        if x is not None:
            plt.scatter(x[:,0], x[:,1], c = 'orange', s = 50)
        if self.dim == 2:
            for r in self.regions:
                p = r.points()
                plt.plot(p[:,0], p[:,1], c = 'black')
                if r.p is not None:
                    plt.scatter(r.p[0], r.p[1], c = 'green', s = 10)
            plt.show()


    def __str__(self):
        s = '# domain\n'
        s += f'dim: {self.dim}\n'
        s += f'll: {self.ll}\n'
        s += f'ur: {self.ur}\n'
        s += f'regions: {len(self.regions)}\n'
        return(s)
    

