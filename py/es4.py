import numpy as np

import numpy.linalg as npl

import scipy.linalg  as spl

import matplotlib.pyplot as plt

A = np.array([[6, 63, 662.2],[63, 662.2, 6967.8],[662.2, 6967.8, 73393.5664]])

b = np.array([1.1, 2.33, 1.7])

numeroDiCondizionamento = npl.cond(A, np.inf)

x = spl.solve(A,b)

print(f'Vettore soluzione del sistema: {x}')
    
print(f'{numeroDiCondizionamento=}')

Aperturbata = A.copy()

Aperturbata[0, 0] = A[0, 0] + 0.01

print(f'{Aperturbata=}')

x_perturbato = spl.solve(Aperturbata, b)

print(f'Vettore soluzione del sistema perturbato: {x_perturbato}')

erroreRelativoDati = npl.norm(A - Aperturbata, np.inf) / npl.norm(A, np.inf)

print("Errore relativo sui dati  in percentuale ", erroreRelativoDati *100,"%")

erroreRelativoSoluzione = npl.norm(x_perturbato - x, np.inf) / npl.norm(x, np.inf)

print("Errore relativo sulla soluzione  in percentuale ", erroreRelativoSoluzione *100,"%")
