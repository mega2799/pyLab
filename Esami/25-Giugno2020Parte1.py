import numpy as np

from numpy.linalg.linalg import det 

import sympy as sym 

import matplotlib.pyplot as plt 

import numpy.linalg as npl 

# def 

A = np.array([ [10, -4, 4, 0], [-4, 10, 0, 2], [4, 0, 10, 2], [0, 2, 2, 0]],dtype=float)

B = np.array([[5, -2, 2, 0], [-2, 5, 0, 1], [2, 0, 5, 1], [0, 1, 1, 5]],dtype=float)   

# a) 

# fattorizzazione di cholesky dovrebbero essere simmetriche e definita positive

# matrice simmetrica risulta uguale alla sua trasposta 

def simm(M):
    assert(np.all(M == M.T)) # np.all controlla che tutti i valori siano True

simm(A)

simm(B)

# definita positiva 

assert(np.all(npl.eigvals(A)))

assert(np.all(npl.eigvals(B)))

#b 

# fattorizzazione LU tutti i determinanti dei minori principali sono maggiori di zero 

def fatt(M):
    for n in range(len(M)):
        assert(np.all(npl.det(M[1:n-1, 1:n-1])))

fatt(A)

fatt(B)

# esiste fattLU per entrambe 

def LU_nopivot(A):
    """
    % Fattorizzazione PA=LU senza pivot   versione vettorizzata
    In output:
    L matrice triangolare inferiore
    U matrice triangolare superiore
    P matrice identità
    tali che  LU=PA=A
    """
    # Test dimensione
    m,n=A.shape
   
    flag=0;
    if n!=m:
      print("Matrice non quadrata")
      L,U,P,flag=[],[],[],1 
      return P,L,U,flag
  
    P=np.eye(n);
    U=A.copy();
 # Fattorizzazione
    for k in range(n-1):
       #Test pivot 
          if U[k,k]==0:
            print('elemento diagonale nullo')
            L,U,P,flag=[],[],[],1 
            return P,L,U,flag

  #     Eliminazione gaussiana
          U[k+1:n,k]=U[k+1:n,k]/U[k,k]                                   # Memorizza i moltiplicatori	  
          U[k+1:n,k+1:n]=U[k+1:n,k+1:n]-np.outer(U[k+1:n,k],U[k,k+1:n])  # Eliminazione gaussiana sulla matrice
     
  
    L=np.tril(U,-1)+np.eye(n)  # Estrae i moltiplicatori 
    U=np.triu(U)           # Estrae la parte triangolare superiore + diagonale
    return P,L,U,flag

# d) 

P,L,U,flag = LU_nopivot(A)

detA = npl.det(A)

invDetA = 1/ detA

print(np.prod(np.diag(U)))

print(detA)