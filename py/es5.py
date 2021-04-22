import numpy as np

import scipy.linalg as spl

import numpy.linalg as npl

n = 4 

A = spl.hilbert(n) # Matrice di Hilbert ordine 4

b = np.array([1, 1, 1, 1])

x = spl.solve(A, b)

print(f'Vettore Soluzione: {x}')

perturbazione = np.array([1, -1, 1, -1])

deltaB = b.copy()

deltaB = deltaB * perturbazione * 0.01

print(deltaB)

xPerturbato = spl.solve(A, b + deltaB)

print(f'Vettore Soluzione perturbato: {xPerturbato}')

err_dati = npl.norm(deltaB,np.inf)/npl.norm(b,np.inf)  
print("Errore relativo sui dati  in percentuale", err_dati*100,"%")

err_rel_sol = npl.norm(x-xPerturbato,np.inf)/npl.norm(x,np.inf) 
print("Errore relativo sulla soluzione  in percentuale", err_rel_sol*100,"%")
