import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import math 

from res.funzioni_Interpolazione_Polinomiale import InterpL, plagr

def zeri_Cheb(n):
    x=np.zeros((n+1,))
    for k in range(n+1):
        x[k]=np.cos(((2*k+1)/(2*(n+1))*math.pi))
    return x
# Definisco funzione
a = -1 

b = 1 

x = sym.symbols('x')

fx = 1 / (1 + 900 * x)

f = lambdify(x, fx, np)

# a) 

xx = np.linspace(a, b, 200)

# Cheby 
for i in range (5, 35, 5):
    checbX = zeri_Cheb(i) # trovo nodi in intervallo 5 : 5 : 30
    solY = f(checbX) # valore dei nodi 
    polC = InterpL(checbX, solY, xx) # polinomio interpolante

    # b) 
    rc = np.abs(f(xx) - polC) # grafico di ambigua utlita'.....

# Newton
for n in range (5, 35, 5):
    nodes = np.linspace(a, b, n + 1) # nodi equispaziati fino ad n + 1
    solY = f(nodes)
    polE = InterpL(nodes, solY, xx) # polinomio interpolante
    
    # b) 
    re = np.abs(f(xx) - polE) # grafico di ambigua utlita'.....
