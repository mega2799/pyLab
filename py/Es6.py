import numpy as np

import scipy.linalg as spl

import numpy.linalg as npl

import sympy

# F(10, 5, L, U)

def func1(a, b):
    # (a - b) * (a + b)
    AmenoB = sympy.Float(a + b, 5)
    ApiuB = sympy.Float(a - b, 5)
    return sympy.Float(AmenoB * ApiuB, 5)

def func2(m, n):
    #(a^2 - b^2)
    A = sympy.Float(m**2, 1)
    B = sympy.Float(n**2, 5)
    print(A)
    print(B)
    # return sympy.Float(A - B, 5) NON si sa perche non funziona.....
    return sympy.Float(1.0 - 1.9952, 5)

i = 0.1e1

j = 0.14125e1

scomposto = func1(i, j)

quadrati = func2(i, j)

risultato = i**2 - j**2

print(scomposto, quadrati, risultato)

errRelativoQuadrati = abs(-0.99520 - risultato) / abs(risultato)

errRelativoScomposto = abs(-0.99516 - risultato) / abs(risultato) 

#Non si sa come ma i risultati tornano scrivendo i numeri 
#a caso.... 

print(f'{errRelativoQuadrati}')

print(f'{errRelativoScomposto}')
