import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

import numpy.linalg as npl

from scipy.linalg import solve

x = sym.symbols('x')

gx = lambda c: (x * (x**2 + 3 *c)) / (3 * x**2 + c)

# c)  

def iterazione(gname, x0, tolx, nmax):
    xk=[]
    xk.append(x0)
    x1 = gname(x0)
    d = x1 - x0
    xk.append(x1)
    it = 1
    while it < nmax and abs(d) > tolx * abs(1):
        x0 = x1
        x1 = gname(x0)
        d = x1 - x0
        it += 1
        xk.append(x1)

    if it == nmax :
        print('raggiunto max')
    return x1, it, xk 

# d) 
x0 = 2

vettc = [1/5, 1/6, 1/7]

sol = []

g1 = lambdify(x, gx(vettc[0]), np)

g2 = lambdify(x, gx(vettc[1]), np)

g3 = lambdify(x, gx(vettc[2]), np)

sol1 = iterazione(g1, x0, 1e-6, 100)

sol2 = iterazione(g2, x0, 1e-6, 100)

sol3 = iterazione(g3, x0, 1e-6, 100)

# e) ordine di convergenza si ha | dgx(x1) | < 1 o funzione, bho non l ho capito 

dg1 = sym.diff(gx(vettc[0]),x,  1)

dg2 = sym.diff(gx(vettc[1]),x,  1)

dg3 = sym.diff(gx(vettc[2]),x,  1)

df1 = lambdify(x, dg1, np)

df2 = lambdify(x, dg2, np)

df3 = lambdify(x, dg3, np)

print(abs(df1(sol1[0])) < 1)

print(abs(df2(sol2[0])) < 1)

print(abs(df3(sol3[0])) < 1)

def stima_ordine(xk, iterazioni):
    p =[]
    for k in range(iterazioni -3):
        p.append(np.log(abs(xk[k+2] - xk[k+3])/ abs(xk[k+1] - xk[k+2])) / np.log(abs(xk[k+1]-xk[k+2])/abs(xk[k]-xk[k+1])))
    ordine = p[-1]
    return ordine 

print(stima_ordine(sol1[2], sol1[1]))

print(stima_ordine(sol2[2], sol2[1]))

print(stima_ordine(sol3[2], sol3[1]))

