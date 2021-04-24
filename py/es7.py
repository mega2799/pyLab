"""
Calcolo exp(x) con serie di Taylor troncata in accordo a test
dell'incremento con x campionato in [a,b]
"""

import funzioniExp 

import numpy as np

import matplotlib.pyplot as plt

a = -10

b = 10

numCampioni = 10000

numList = np.linspace(a, b, numCampioni) # Returns num evenly spaced samples, calculated over the interval [start, stop].

expEsatta = np.exp(numList)

expApprossimata = np.zeros((numCampioni,)) # Return a new array of given shape and type, filled with zeros.

nIndexes = np.zeros((numCampioni,))

for i in range(numCampioni):
    expApprossimata[i], nIndexes[i]  = funzioniExp.esp_taylor_1(numList[i])

erroreRelativo = np.abs(expApprossimata - expEsatta) / np.abs(expEsatta)

plt.plot(numList, expApprossimata, '-b', numList, expEsatta, 'r--')

plt.title('Approssimazione esponenziale focon serie di Taylor troncata')

plt.legend(['expApprossimata','expEsatta'])

plt.show()


""" Coseh """


plt.plot(numList, erroreRelativo)

plt.title("Errore relativo scala cartesiana")

plt.show()


plt.plot(numList, erroreRelativo)

plt.yscale("log")

plt.title("Errore relativo scala semi-logaritimica")

plt.show()



plt.plot(numList, nIndexes)

plt.title('Indice n')

plt.show()

'''
--------------------------------------------------------------------------
come migliorare andamento errore relativo
--------------------------------------------------------------------------
'''

for i in range(numCampioni) :
    if numList[i]>=0:
        expApprossimata[i], nIndexes[i] = funzioniExp.esp_taylor_1(numList[i])
    else:
        expApprossimata[i], nIndexes[i] = funzioniExp.esp_taylor_2(numList[i])


err_rel_2 = np.abs(expApprossimata - expEsatta)/np.abs(expEsatta)

plt.plot(numList, err_rel_2)

plt.yscale("log")

plt.title('Errore relativo Algoritmo Migliorato - scala semilogaritmica')

plt.show()