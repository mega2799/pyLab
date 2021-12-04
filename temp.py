import math

import numpy as np

import sympy as sym

import res.funzioniZeri as funzioniZeri

import matplotlib.pyplot as plt

from sympy.utilities.lambdify import lambdify

tolleranzaX = 1e-6

tolleranzaF = 1e-5

x = sym.symbols('x') 

fx = sym.atan(x) 

deltaF = sym.diff(fx, x, 1) 

#Trasformo in numeriche la funzione e la sua derivata

f = lambdify(x, fx, np)

df = lambdify(x, deltaF, np)

insiemeNum = np.linspace(-10, 10, 100)

plt.plot(insiemeNum, 0 * insiemeNum, insiemeNum, f(insiemeNum), 'r-') #asse X 

plt.show()

nmax = 500


#Considero come iterato iniziale per Newton: x0=1.2: il metodo converge
x0=1.2

xNew, itNew, xkNew=funzioniZeri.newton(f, df, x0, tolleranzaX, tolleranzaF, nmax)

print('X0= {:e} ,  zero Newton= {:e} con {:d} iterazioni \n'.format(x0, xNew, itNew))

#Considero come iterato iniziale per Newton: x0=1.4: il metodo non converge  
x0=1.4

xNew, itNew, xkNew=funzioniZeri.newton(f, df, x0, tolleranzaX, tolleranzaF, nmax)

print('X0= {:e} ,  zero Newton= {:e} con {:d} iterazioni \n'.format(x0, xNew, itNew))