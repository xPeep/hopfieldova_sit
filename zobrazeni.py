import matplotlib.pyplot as plt
import numpy as np


def show(vzor):
    # zobrazeni prislusneho vzoru
    plt.imshow(vzor, cmap='gray')
    plt.show()
    plt.pause(1)


def noise(vzor, prav):
    size = vzor.shape
    for ind1 in range(size[0]):
        for ind2 in range(size[1]):
            if np.random.rand() <= prav:
                vzor[ind1, ind2] = -vzor[ind1, ind2]

    return vzor
