import numpy as np
from zobrazeni import show
from zobrazeni import noise
from hopfield import Hopfield

vzor1 = np.genfromtxt('vzory2\\a.csv', delimiter=';')
vzor2 = np.genfromtxt('vzory2\\b.csv', delimiter=';')
vzor3 = np.genfromtxt('vzory2\\c.csv', delimiter=';')
vzor4 = np.genfromtxt('vzory2\\d.csv', delimiter=';')
print(vzor1.shape)
hop = Hopfield(125 * 100)

hop.train_iter(vzor1.flatten())
hop.train_iter(vzor2.flatten())
hop.train_iter(vzor3.flatten())
hop.train_iter(vzor4.flatten())

show(vzor1)
y_noise = noise(vzor1, 0.39)
show(y_noise)
y_noise = hop.equip_iter(y_noise.flatten())
y_noise = np.reshape(y_noise, (125, 100))
show(y_noise)

y_noise = hop.equip_iter(y_noise.flatten())
y_noise = np.reshape(y_noise, (125, 100))
show(y_noise)
