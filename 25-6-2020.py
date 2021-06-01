import numpy as np

from src.Sistemi_lineari import LU_nopivot

A = np.array([[10, -4, 4, 0], [-4, 10, 0, 2], [4, 0, 10, 2], [0, 2, 2, 0]], dtype=float)

B = np.array([[5, -2, 2, 0], [-2, 5, 0, 1], [2, 0, 5, 1], [0, 1, 1, 5]],dtype=float)

# Stabilire se A B ammettono fattorizzazione di Cholesky

# Stabilire se A B ammettono fattorizzazione LU senza pivoting

#risp: Matrici quadrate e elementi sulla diagonale tutti diversi da 0 della matrice triangolare

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
