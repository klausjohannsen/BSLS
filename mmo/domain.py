# libs
import numpy as np
import numpy.linalg as la
from copy import deepcopy as copy
import matplotlib.pyplot as plt

# fcts

# classes
class Region:
    def __init__(self, ll = None, ur = None, p = None, penalty = 0):
        assert(ll is not None)
        assert(ur is not None)
        self.ll = ll
        self.ur = ur
        assert(self.ll.shape[0] == self.ur.shape[0])
        self.dim = self.ll.shape[0]
        self.p = p
        self.midpoint = 0.5 * (self.ll + self.ur)
        self.l = self.ur - self.ll
        self.closed_corner_polygon = np.array([ self.ll[0], self.ll[1], self.ur[0], self.ll[1], self.ur[0], self.ur[1], self.ll[0], self.ur[1], self.ll[0], self.ll[1] ]).reshape(5,2)
        self.volume = np.prod(self.l)
        self.penalty = 0

        # score
        self.score = self.volume * (0.4 ** self.penalty)
        if self.p is not None: self.score *= 0.5

    def __llur_contains(self, ll, ur, x):
        c = True
        for k in range(x.shape[0]):
            if x[k] < ll[k]: c = False
            if x[k] >= ur[k]: c = False
        return(c)

    def contains(self, x):
        return(self.__llur_contains(self.ll, self.ur, x))

    def __split_parameters(self, x):
        assert(self.contains(x))
        d = np.inf
        for k in range(self.dim):
            eta = (x[k] - self.ll[k]) / self.l[k]
            if np.abs(eta - 0.5) < d:
                d = np.abs(eta - 0.5)
                k_best = k
                eta_best = eta
        return(k_best, eta_best)

    def __longest_axis(self):
        return(np.argmax(self.l))

    def bisect(self, p = None):
        if self.p is None and p is None:
            # empty region, no point inserted: split equally into two, penalized
            ll_1 = copy(self.ll)
            ur_1 = copy(self.ur)
            ll_2 = copy(self.ll)
            ur_2 = copy(self.ur)
            axis = self.__longest_axis()
            mp = 0.5 * (self.ll[axis] + self.ur[axis])
            ur_1[axis] = mp
            ll_2[axis] = mp

            # penalized
            r1 = Region(ll = ll_1, ur = ur_1, p = None, penalty = self.penalty + 1)
            r2 = Region(ll = ll_2, ur = ur_2, p = None, penalty = self.penalty + 1)
            return(r1, r2)

        if self.p is not None and p is None:
            # non-empty region, no point inserted: split equally into two, penalized
            ll_1 = copy(self.ll)
            ur_1 = copy(self.ur)
            ll_2 = copy(self.ll)
            ur_2 = copy(self.ur)
            axis = self.__longest_axis()
            mp = 0.5 * (self.ll[axis] + self.ur[axis])
            ur_1[axis] = mp
            ll_2[axis] = mp

            # check where point lies
            c1 = self.__llur_contains(ll_1, ur_1, self.p)
            c2 = self.__llur_contains(ll_2, ur_2, self.p)
            assert(c1 ^ c2)

            # penalized
            p1 = self.p if c1 else None
            p2 = self.p if c2 else None
            r1 = Region(ll = ll_1, ur = ur_1, p = p1, penalty = self.penalty + 1)
            r2 = Region(ll = ll_2, ur = ur_2, p = p2, penalty = self.penalty + 1)
            return(r1, r2)

        if self.p is None and p is not None:
            # empty region, point inserted
            ll_1 = copy(self.ll)
            ur_1 = copy(self.ur)
            ll_2 = copy(self.ll)
            ur_2 = copy(self.ur)
            axis = self.__longest_axis()
            mp = 0.5 * (self.ll[axis] + self.ur[axis])
            ur_1[axis] = mp
            ll_2[axis] = mp

            # check where point lies
            c1 = self.__llur_contains(ll_1, ur_1, p)
            c2 = self.__llur_contains(ll_2, ur_2, p)
            assert(c1 ^ c2)

            # not penalized
            p1 = p if c1 else None
            p2 = p if c2 else None
            r1 = Region(ll = ll_1, ur = ur_1, p = p1, penalty = self.penalty)
            r2 = Region(ll = ll_2, ur = ur_2, p = p2, penalty = self.penalty)
            return(r1, r2)

        if self.p is not None and p is not None:
            # non-empty region, point inserted
            ll_1 = copy(self.ll)
            ur_1 = copy(self.ur)
            ll_2 = copy(self.ll)
            ur_2 = copy(self.ur)
            axis, eta = self.__split_parameters(0.5 * (self.p + p))
            ur_1[axis] = self.ll[axis] + eta * self.l[axis]
            ll_2[axis] = ur_1[axis]

            # check where points lies
            self_c1 = self.__llur_contains(ll_1, ur_1, self.p)
            self_c2 = self.__llur_contains(ll_2, ur_2, self.p)
            assert(self_c1 ^ self_c2)
            c1 = self.__llur_contains(ll_1, ur_1, p)
            c2 = self.__llur_contains(ll_2, ur_2, p)
            assert(c1 ^ c2)
            if self_c1 == c1 or self_c2 == c2:
                print('ll', self.ll)
                print('ur', self.ur)
                print('ll_1', ll_1)
                print('ur_1', ur_1)
                print('ll_2', ll_2)
                print('ur_2', ur_2)
                print('self.p', self.p)
                print('p', p)
                exit()

            # not penalized
            p1 = p if c1 else self.p
            p2 = p if c2 else self.p
            r1 = Region(ll = ll_1, ur = ur_1, p = p1, penalty = self.penalty)
            r2 = Region(ll = ll_2, ur = ur_2, p = p2, penalty = self.penalty)
            return(r1, r2)

    def __str__(self):
        s = '# region\n'
        s += f'dim: {self.dim}\n'
        s += f'll: {self.ll}\n'
        s += f'ur: {self.ur}\n'
        s += f'p: {self.p}\n'
        s += f'penalty: {self.penalty}\n'
        s += f'score: {self.score}\n'
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

    def get_top_region(self):
        assert(len(self.regions) > 0)
        score_max = -np.inf
        for region in self.regions:
            if region.score > score_max:
                score_max = region.score
                region_max = region
        return(region_max)

    def get_region_with_point(self, x):
        region = None
        for r in self.regions:
            if r.contains(x):
                assert(region is None)
                region = r 
        return(region)

    def replace(self, regions_in = None, regions_out = None):
        for r in regions_out:
            self.regions.remove(r)
        self.regions += regions_in

    def solutions(self):
        s = []
        for r in self.regions:
            if r.p is not None:
                s += [r.p]
        return(np.vstack(s))

    def plot(self, x = None):
        if x is not None:
            plt.scatter(x[:,0], x[:,1], c = 'orange', s = 50)
        if self.dim == 2:
            for r in self.regions:
                p = r.closed_corner_polygon
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
    

