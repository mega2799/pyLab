import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

import numpy.linalg as npl

from scipy.linalg import solve

# def 

A = np.array([[1, 2, 4, 0, 0], [2, 6, 0, 0, 0],[4, 0, 3, 1, 0], [0, 0, 1, -1/45, 5],[0, 0, 0, 5, 1]], dtype=float)

# a) 

# determinanti delle sottomatrici minori principali sono tutti diversi da zero 

n, m = A.shape

for i in range(n):
    assert(np.all(npl.det(A[1:i-1, 1:i-1])))


# LU ammessa..

# b) 

def swapRows(A,k,p):
    A[[k,p],:] = A[[p,k],:]
    

def LU_pivot(A):
    """
    % Fattorizzazione PA=LU con pivot 
    In output:
    L matrice triangolare inferiore
    U matrice triangolare superiore
    P matrice di permutazione
    tali che  PA=LU
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
       #Scambio di righe nella matrice U e corrispondente scambio nella matrice di permutazione per
       # tenere traccia degli scambi avvenuti
       
       #Fissata la colonna k-esima calcolo l'indice di riga p a cui appartiene l'elemento di modulo massimo a partire dalla riga k-esima
          p = np.argmax(abs(U[k:n,k])) + k
          if p != k:
              swapRows(P,k,p)
              swapRows(U,k,p)

  #     Eliminazione gaussiana
          U[k+1:n,k]=U[k+1:n,k]/U[k,k]                                   # Memorizza i moltiplicatori	  
          U[k+1:n,k+1:n]=U[k+1:n,k+1:n]-np.outer(U[k+1:n,k],U[k,k+1:n])  # Eliminazione gaussiana sulla matrice
     
  
    L=np.tril(U,-1)+np.eye(n)  # Estrae i moltiplicatori 
    U=np.triu(U)           # Estrae la parte triangolare superiore + diagonale
    return P,L,U,flag

       
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


P, L, U, flag = LU_pivot(A)

# c) 

# posso calcolarlo velocemente utilizzando la U triangolare inferiore, moltiplicando gli elementi della diagonale 

detA = np.prod(np.diag(U))

# impiega n = 5 moltiplicazioni

# d) 

v = np.array([ [1.2], [3.5], [1.66], [78.0], [44.9]]) # vettore casuale ma scritto cosi in modo che il npl.dot non rompa...

x = solve(A, v)

res = npl.det( A + np.dot(v, v.T))

v = np.array([ 1.2, 3.5, 1.66, 78.0, 44.9]) # riscritto perche altrimenti non c'era corrispondenza per il prodotto vettoriale

result = (np.dot(v, x) + 1) * detA

# entrambi i numeri portano allo stesso risultato, il secondo potrebbe risultare piu efficente perche il detA 
# è stato facilmente calcolato al punto c) e poi ce un semplice prodotto ed un vettoriale