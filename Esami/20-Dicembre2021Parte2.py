import numpy as np

import sympy as sym 

import matplotlib.pyplot as plt 

from sympy.utilities import lambdify

import scipy.linalg as spl

import math 

def SimpComp(fname, a, b, n):
    h = (b-a)/(2*n)
    nodi = np.arange(a, b + h, h)
    f = fname(nodi)
    I = (f[0] + 2*np.sum(f[2:2*n:2]) + 4 * np.sum(f[1:2*n:2]) + f[2*n]) * h/3 
    return I 

def simptoll(fun, a,b, toll):
    Nmax = 2048 
    err = 1
    N = 1
    IN = SimpComp(fun, a, b, N)
    while N <= Nmax and err > tol:
        N = 2* N 
        I2N = SimpComp(fun, a, b, N)
        err = abs(IN - I2N)/15 
        IN = I2N 
        if N > Nmax:
            print('raggiunto max')
            N = 0
            IN = []
            return IN, N
    return IN, N
    

def trapComp(fname, a, b, n):
    h = (b-a)/n 
    nodi = np.arange(a, b + h, h)
    f = fname(nodi)
    I = (f[0] + 2 * np.sum(f[1:n] + f[n])) * h/2 
    return I 

def traptoll(fun, a, b, toll):

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
    return IN, N
    

# def 
x = sym.symbols('x')

a = -2 

b = 2

f0x = x**3 + 1 

f1x = x**3 - 2*x**2 + 1 

f0 = lambdify(x, f0x, np)

f1 = lambdify(x, f1x, np)

tol = 10e-6 
# a) 
t0, intervalli0  = (traptoll(f0, a, b, tol))

t1, intervalli1 =(traptoll(f1, a, b, tol))

# e) 

print('TRAPEZI')

print('sol: {}, ottentuta con {} intervalli'.format(t0, intervalli0))

print('sol: {}, ottentuta con {} intervalli'.format(t1, intervalli1))

# c)
s0, intervalli0  = (simptoll(f0, a, b, tol))

s1, intervalli1 =(simptoll(f1, a, b, tol))

print('SIMPSON')

print('sol: {}, ottentuta con {} intervalli'.format(s0, intervalli0))

print('sol: {}, ottentuta con {} intervalli'.format(s1, intervalli1))

# con simpson arrivo ad una stima mentre con trapezi eccedo il numero max di terazione