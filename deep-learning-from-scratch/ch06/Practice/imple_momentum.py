import numpy as np


class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, gards):
        self.v = {}
        for key, val in params.items():
            self.v[key] = np.zeros_like(val)
            params[key] += self.v[key]