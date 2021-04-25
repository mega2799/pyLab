import math

import numpy as np

import sympy as sym

import funzioniZeri

import matplotlib.pyplot as plt

from sympy.utilities.lambdify import lambdify

tolleranzaX = 1e-8

tolleranzaF = 1e-8

a = 3/5*math.pi

b = 37/25*math.pi

x = sym.symbols('x') 

fx = sym.tan(x) - x

deltaF = sym.diff(fx, x, 1) 

#Trasformo in numeriche la funzione e la sua derivata

f = lambdify(x, fx, np)

df = lambdify(x, deltaF, np)

insiemeNum = np.linspace(a, b, 100)

plt.plot(insiemeNum, f(insiemeNum), 'r-')

plt.plot(insiemeNum, 0 * insiemeNum, insiemeNum, f(insiemeNum), 'r-') #asse X 

plt.show()


