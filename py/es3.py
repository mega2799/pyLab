import numpy as np

import numpy.linalg as npl

import scipy.linalg  as spl

import matplotlib.pyplot as plt

A = np.array([ [3, 5], [3.01, 5.01] ]) # matrice dei coefficenti

b = np.array([10, 1]) # vettore dei risultati

x = spl.solve(A, b) # Soluzione del sistema

print(f'Vettore soluzione sistema: {x} %')


deltaA = np.array([ [0, 0], [0.01, 0]])

erroreDati = npl.norm(deltaA, np.inf) / npl.norm(A, np.inf) # dati perturbati/
                                                            # dati esatti

print(f'Errore relativo sui dati: {erroreDati * 100} %')

x1 = spl.solve(A + deltaA, b) # soluzione del sistema perturbato

print(f'Vettore soluzione sistema perturbato: {x1}')
