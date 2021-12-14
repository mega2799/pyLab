import math

import numpy as np 

import sympy as sym 

import numpy.linalg as npl 

import matplotlib.pyplot as plt

from scipy.optimize import fsolve

from sympy.utilities import lambdify 

# def funzione 

x = sym.symbols('x')

fx = sym.exp(x) - 4 * x**2 

f = lambdify(x, fx, np)

a = -1 

b = 5
 
# a) 

xx = np.linspace(a, b, 100)

plt.plot(xx, f(xx))

plt.axhline(y=0)

plt.axvline(x=0)

plt.show()

# il grafico sembra avere 3 radici reali, di cui 1 negativa e 2 positive 

# le 3 radice sono negli intervalli [-1, 0], [0, 1] e [4, 5] con fsolve trovo le intersezioni con asse X 
# uso come x0 valori che rientrano dentro quell intervallo 

x0 = -.8
alfa1 = fsolve(f, x0)
x0 = 0.5
alfa2 = fsolve(f, x0) 
x0 = 4.3
alfa3 = fsolve(f, x0) 

print(alfa1, alfa2, alfa3)

# b) 

gx = 0.5 * sym.exp(x/2)

g = lambdify(x, gx, np)

plt.plot(xx, g(xx))

plt.axhline(y=0)

plt.axvline(x=0)

plt.show()

# dal grafico non credo che gx possa essere usata per determinare tutte le radici di fx 

# c) 

def iterazione(gname,x0,tolx,nmax):
        xk=[]
        xk.append(x0)
        x1=gname(x0)
        d=x1-x0
        xk.append(x1)
        it=1
        while it<nmax and  abs(d)>=tolx*abs(x1) :
            x0=x1
            x1=gname(x0)
            d=x1-x0
            it=it+1
            xk.append(x1)
           
    
        if it==nmax:
            print('Raggiunto numero max di iterazioni \n')
        
        return x1, it,xk

# d) 

toll = 1e-6 

x, it, xk = iterazione(g, .5, toll, 100)

print('iterazioni= {:d}, soluzione={:e} \n\n'.format(it,x))

# risulta convergente a 0.7 quindi vicino alla soluzione

x, it, xk = iterazione(g, 4.5, toll, 100)

print('iterazioni= {:d}, soluzione={:e} \n\n'.format(it,x))

# non risulta convergente la soluzione non esiste...

# e) mi scoccia......