import sympy as sym

import numpy as np

#F(10, 2, -3, 3) 

a = 0.96e-1

b = 0.99e-1

#sympy.Float(<operazione>, 2) arrotonda il risultato a 2 cifre siginificative

somma = sym.Float(a + b, 2)

pMedio = sym.Float(somma * 0.5, 2)

print(f'(a + b)/2 = {pMedio}')

differenza = sym.Float(b - a, 2)

somma = sym.Float(a + differenza,2) 

pMedio = sym.Float(somma + 0.5, 2)

print(f'(a + (b -a))/2 = {pMedio}')
