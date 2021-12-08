import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

import numpy.linalg as npl

A = lambda i : np.array([[1, -1, 2], [-1, 6, 1], [2, 1, i]])

# a) 

normaInf = [] 

for i in range(6, 11):
    normaInf.append(npl.norm(A(i), np.inf))

# b)
print((normaInf))

# il valore minimo lo ho con alpha uguale a 6 e vale 9.0

# c) 

print(npl.norm(A(6), 1)) # anche la norma 1 vale 9.0

# La matrice che ha norma 1 ed infinito uguali è simmetrica
# quindi A(6) è simmetrica 

# d) 

# Una matrice ammette fattorizzazione di cholesky se simmetrica e definita positiva 

# so gia che è simmetrica dal punto c, verifico che è definita positiva trovando gli autovalori
# che devono essere tutti > 0 

print(npl.eigvals(A(6)))

# sono tutti positivi, quindi la matrice ammette la fattorizzazione di cholesky :^ )