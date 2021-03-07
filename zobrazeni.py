import matplotlib.pyplot as plt
import numpy as np

def show(vzor):
    # zobrazeni prislusneho vzoru
    plt.imshow(vzor, cmap='gray')
    plt.show()
    plt.pause(1)


