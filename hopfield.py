import numpy as np


class Hopfield:
    def __init__(self, n_inputs):
        self.w = np.zeros((n_inputs, n_inputs))

    def train_iter(self, vzor):
        n = len(vzor)
        # delta_w = np.zeros((n, n))
        # for ind1 in range(n):
        #    for ind2 in range(n):
        #        if ind1 != ind2:
        #            delta_w[ind1, ind2] = vzor[ind1] * vzor[ind2]
        delta_w = np.tensordot(vzor, vzor, axes=0)
        self.w = self.w + delta_w

    def equip_iter(self, x):
        n = len(x)
        # y_a = np.zeros((n, 1))
        # for ind1 in range(n):
        #    for ind2 in range(n):
        #        y_a[ind1] = y_a[ind1] + self.w[ind1, ind2] * x[ind2]

        y_a = self.w.dot(x)
        y = np.zeros((n, 1))
        for ind in range(n):
            if y_a[ind] >= 0:
                y[ind] = 1
            else:
                y[ind] = -1

        return y
