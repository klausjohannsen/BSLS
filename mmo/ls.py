import numpy as np
from cmaes import CMA

# cma
class Cma:
    def __init__(self, x0 = None, f = None, sigma = None, max_gen = 10**20):
        optimizer = CMA(mean = x0, sigma = sigma)
        y_best = np.inf
        n_fct_eval = 0
        for gen in range(max_gen):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                y = f(x)
                n_fct_eval += 1
                solutions.append((x, y))
                if y < y_best:
                    x_best = x
                    y_best = y
            optimizer.tell(solutions)
            if optimizer.should_stop():
                break

        self.n_fct_eval = n_fct_eval
        self.x = x_best
        self.y = y_best
        self.n_gen = gen

    def __str__(self):
        s = '## cmaes\n'
        s += f'x = {self.x}\n'
        s += f'y = {self.y}\n'
        s += f'n_gen = {self.n_gen}\n'
        s += f'n_fct_eval = {self.n_fct_eval}'
        return(s)







