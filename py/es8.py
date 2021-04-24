import numpy as np

import math

import matplotlib.pyplot as plt

k = np.arange(0, -21, -1) #Return evenly spaced values within a given interval, Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop).
# array di numeri negativi [0,-1 .....-20]
h=10.0**k
#array di numeri piccoli
derivataEsatta = math.cos(1)

x = 1

rapportoIncrementale = (np.sin(x + h) - np.sin(x)) / h

erroreRelativo = np.abs(rapportoIncrementale - derivataEsatta) / np.abs(derivataEsatta)


plt.plot(h, erroreRelativo, 'b-', h, h, 'r:')

plt.xscale("log")

plt.yscale("log")

plt.legend(['Errore relativo', 'Incremento'])