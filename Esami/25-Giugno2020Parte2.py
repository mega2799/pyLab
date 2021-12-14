import math
import numpy as np
from numpy.linalg.linalg import det 

import sympy as sym 

import matplotlib.pyplot as plt 

import sympy.utilities.lambdify as lambdify

import numpy.linalg as npl 

# def 

fx = lambda x : x - np.sqrt( x - 1)

a = 1 

b = 3 

# creo nodi equispaziati 

xx = np.linspace(a, b, 200)

# grado 3 

x = np.linspace(a, b, 3 + 1)

y = fx(x)

def plagr(xnodi,k):
    """
    Restituisce i coefficienti del k-esimo pol di
    Lagrange associato ai punti del vettore xnodi
    """
    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if k==0:
       xzeri=xnodi[1:n]
    else:
       xzeri=np.append(xnodi[0:k],xnodi[k+1:n])
    
    num=np.poly(xzeri) 
    den=np.polyval(num,xnodi[k])
    
    p=num/den
    
    return p



def InterpL(x, f, xx):
     """"
        %funzione che determina in un insieme di punti il valore del polinomio
        %interpolante ottenuto dalla formula di Lagrange.
        % DATI INPUT
        %  x  vettore con i nodi dell'interpolazione
        %  f  vettore con i valori dei nodi 
        %  xx vettore con i punti in cui si vuole calcolare il polinomio
        % DATI OUTPUT
        %  y vettore contenente i valori assunti dal polinomio interpolante
        %
     """
     n=x.size
     m=xx.size
     L=np.zeros((n,m))
     for k in range(n):
        p=plagr(x,k)
        L[k,:]=np.polyval(p,xx)
    
    
     return np.dot(f,L)

# calcolo il polinomio 

pol = InterpL(x, y, xx)

print(pol)

# b) 

plt.plot(xx, fx(xx), xx, pol)

plt.legend(['funzione', 'funzione interpolante'])

plt.show()
