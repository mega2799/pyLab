import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

import numpy.linalg as npl

# def 

x = sym.symbols('x')

f = sym.sin(sym.sqrt(x))

a = 0 

b = 1

def trapComp(fname, a, b, n):
    h = (b-a)/n 
    nodi = np.arange(a, b + h, h)
    f = fname(nodi)
    I = (f[0] + 2 * np.sum(f[1:n] + f[n])) * h/2 
    return I 

def tratoll(fun, a, b, toll):

    Nmax = 2048 
    err = 1

    N = 1
    IN = trapComp(fun, a, b, N)

    while(N <= Nmax and err > toll):
        N = 2*N 
        I2N = trapComp(fun,a, b, N)
        err = abs(IN - I2N)/3
        IN = I2N

        if N > Nmax:
            print('raggiunto max')
            N=0
            IN = []
        return IN, N 
# a) 

# credo che sia un integrale che se risolto con il metodo somma/prodotto non si semplifichi e vada, in generale la radice complica la soluzione 


#f1_1 = np.sin(np.sqrt(x)) - np.sqrt(x) 
f1_1 = sym.sin(sym.sqrt(x)) - sym.sqrt(x)

#f1_2 = lambda x : np.sqrt(x) 
f1_2 = sym.sqrt(x) 

xx = np.linspace(a, b, 100)

sol1_1 = sym.integrate(f1_1, (x,a, b))

sol1_2 = sym.integrate(f1_2, (x,a, b))

print("sol1: {} \t sol2: {} \nsol totale: {}" .format(sol1_1, sol1_2, sol1_1 + sol1_2))

f2_1 = sym.sin(sym.sqrt(x)) - sym.sqrt(x) * ( 1 - x / 6)

f2_2 = sym.sqrt(x) * ( 1 - x / 6)

sol2_1 = sym.integrate(f2_1, (x,a, b))

sol2_2 = sym.integrate(f2_2, (x, a, b))

print("sol1: {} \t sol2: {} \nsol totale: {}" .format(sol2_1, sol2_2, sol2_1 + sol2_2))

sol = sym.integrate(f, (x, a, b ))

print(sol)

# ottengo lo stesso risultato, la prima risulta piu facile perche con somma/prodotto dovrebbe risolversi facilmente 

# c) 
fx = lambdify(x, f, np)

toll = 10e-6 

print(tratoll(fx, a,b, toll))

f1x = lambdify(x, f1_1 + f1_2, np)

print(tratoll(f1x, a,b, toll))

f2x = lambdify(x, f2_1 + f2_2, np)

print(tratoll(f2x, a,b, toll))

# le 3 approssimazioni sono identiche 

# d) 

print(2*(math.sin(1) - math.cos(1)))