import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

# Def funzione 

x = sym.symbols('x')

fx = x - 1/3 * sym.sqrt(30 * x -25)

f = lambdify(x, fx, np) 

a = 5/6 

b = 25/6 


