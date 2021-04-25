import numpy as np

import sympy as sym 

import matplotlib.pyplot as plt 

from sympy.utilities.lambdify import lambdify

import funzioniZeri

scelta = input("Scegli quale tra le 3 funzioni\n")

x = sym.symbols('x')

# key : [ funzioneSimbolica, f(0), estremoDX, estremoSX, valore innesco x0, , secondoIterato(secanti) ] 

functions = {
        '1': [sym.exp(-x)-(x+1),0,-1,2,-0.5,-0.3],
        '2': [sym.log(x+3,2)-2,1,-1,2,-0.5,0.5],
        '3': [sym.sqrt(x)-x**2/4, 2**(4/3),1,3,1.8,1.5]
    }

func = functions.get(scelta)

f, aplha, a, b, x0, xm1 = func

deltaF = sym.diff(f, x, 1)

#Rendo numeriche la f e la derivata

fNumerica = lambdify(x, f, np)

deltaFNumerica = lambdify(x, deltaF, np)

print(deltaFNumerica, fNumerica )
