import numpy as np

from src.Sistemi_lineari import LU_nopivot

A = np.array([[10, -4, 4, 0], [-4, 10, 0, 2], [4, 0, 10, 2], [0, 2, 2, 0]], dtype=float)

B = np.array([[5, -2, 2, 0], [-2, 5, 0, 1], [2, 0, 5, 1], [0, 1, 1, 5]],dtype=float)

# Stabilire se A B ammettono fattorizzazione di Cholesky

# Stabilire se A B ammettono fattorizzazione LU senza pivoting

#risp: Se i minori principali della matrice hanno rando massimo allora la fattorizzazione LU si puo applcare

deter = []

for i in range(0,4):
    deter.append(np.linalg.det(A[:i+1, :i+1]))

if np.all(deter!=0):
    print("Matrice A ammette fattorizzazione LU")
    
deter = []

for i in range(0,4):
    deter.append(np.linalg.det(B[:i+1, :i+1]))

if np.all(deter != 0):
    print("Matrice B ammette fattorizzazione LU")
 

def LUsenzaPivoting(M):
    m,n = M.shape
    if m != n:
        print("non sono quadrate") 
        return [],[],[],1

    P = np.eye(n)
    U = M.copy()
    for i in range(n - 1):
        if M[i,i]==0:
            print("elemento nullo")
            return [],[],[],1
        U[i+1:n, i] = U[i+1:n, i] / U[i,i]
        U[i+1:n, i+1:n] = U[i+1:n, i+1:n] - np.outer( U[i+1:n, i], U[i,i+1:n])
    
    L = np.tril(U, -1) + np.eye(n)
    U = np.triu(U)
    return P, L, U, 1

p, l, u, flag = LUsenzaPivoting(A)

pi, li, ui, flag = LU_nopivot(A)

print(l==li,u==ui)

p, l, u, flag = LUsenzaPivoting(B)

p, li, ui, flag = LU_nopivot(B)

print(l==li,u==ui)

# Scrivere uno script che sfrutti l’output dell’algoritmo di fattorizzazione LU senza pivoting per calcolarenella maniera pi`u efficiente possibile sia il determinante diMche il determinante diM^−1

# ho PA = LU => det(A) = det(L,U) 
P, L, U, flag = LUsenzaPivoting(A)

LU = np.dot(L,U)

print("det A " , np.linalg.det(LU))

print("det A^-1 " , 1/np.linalg.det(LU))

P, L, U, flag = LUsenzaPivoting(B)

LU = np.dot(L,U)

print("det B ",np.linalg.det(LU))

print("det B^-1 " ,1/np.linalg.det(LU))

###################################################################### 

from src.funzioni_Interpolazione_Polinomiale import *

import matplotlib.pyplot as plt

def simptoll(fname, a, b, tol):
    Nmax = 2048
    err = 1
    N = 1
    In = simpcomp(fname, a, b, N)
    while N <= Nmax and err > tol:
        N = 2 * N
        I2n = simpcomp(fname, a, b, N)
        err = abs(In - I2n) / 15
        In = I2n
    if N > Nmax:
        return 0, []
    return In, N

def simpcomp(fname, a, b, N):
    h = (b-a)/2*N
    nodi = np.arange(a, b+h, h)
    f = fname(nodi)
    i = (f[0] + 2*np.sum(f[2:2*n:2]) + 4*np.sum(f[1:2*n:2]) +f[2*n]) * (h/3)
    return i

f = lambda x : x - np.sqrt(x-1) 

# Estremi integrazione
a = 1

b = 3

n = 3 # Grado = 3 

# Costruisco il polinomio
x = np.linspace(a, b, n + 1)

y = f(x)

z = np.linspace(a, b, 100) 

# Polinomio interpolatore di Lagrange
pol = InterpL(x, y, z)

plt.plot(z, f(z), z, pol, x, y, 'o')

plt.legend(['Funzione da interpolare','Polinomio interpolatore', 'Nodi di interpolazione'])

plt.show()

# Calcolare con Simpson i valori approssimati di 2 inntegrali 

tol = 1e-5

it1, N1 = simptoll(f, a, b, tol)

it2, N2 = simptoll(pol, a, b, tol)


#I1 ed I2 sono i valori esatti dei due integrali 
I1 = 2.114381916835873
I2 = 2.168048769926493 

errore1 = abs(it1 - IT1)

errore2 = abs(it2 - IT2)

print('Errore integrale funzione f(x)', err1,' Numero di suddivisioni ', N1)

print('Errore integrale del polinomio interpolatore', err2, 'Numero di suddivisioni ',N2)
