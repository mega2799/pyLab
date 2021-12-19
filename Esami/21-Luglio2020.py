import numpy as np

import sympy as sym

import matplotlib.pyplot as plt

from sympy.utilities.lambdify import lambdify

from scipy.optimize import fsolve

# def 

x = sym.symbols('x')

px = sym.tan(1.5*x) - 2 * sym.cos(x) - x*(7-x)

p = lambdify(x, px, np)

fx = sym.tan(1.5*x) - 2 * sym.cos(x) - x*(6-x)

a = -1 

b = 1 

# a) 

xx = np.linspace(a, b, 200)

f = lambdify(x, fx, np)

deltaf = sym.diff(fx, x, 1)

df = lambdify(x, deltaf, np)

# dal grafico so che passa per 0 in 3 punti quindi 

x0 = 0

sol = fsolve(p, x0)

assert(not(abs(df(sol)) < 1)) # la soluzione della funzione INIZIALE! non e' punto fisso.... 

plt.plot(xx, df(xx), sol, 0, 'ro') # graficamente il punto non passa per (1, -1) e la funzione derivata
plt.legend(['funzione f\'(x)', 'soluzione in 0'])
plt.axhline(y=1)
plt.axhline(y=-1)
plt.show()

#b) ha scambiato un po di numeri e viene x = (sym.tan(3.0/2.0*x)-2*sym.cos(x)+x**2)/7, stessa identica funzione

gx = (sym.tan(3.0/2.0*x)-2*sym.cos(x)+x**2)/7 

g = lambdify(x, gx, np)

deltag = sym.diff(gx, x, 1)

dg = lambdify(x, deltag, np)

assert(abs(dg(sol)) < 1) # e' un punto fisso 

plt.plot(xx, dg(xx), sol, 0, 'ro') # graficamente il punto passa per (1, -1) e la funzione derivata
plt.axhline(y=1)
plt.axhline(y=-1)
plt.legend(['funzione g\'(x)', 'soluzione in 0'])
plt.show()

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

#d) 
x1, it, xk = iterazione(g, x0, 1e-7, 500)

print('sol: ', x1)


# e) 
def stima_ordine(xk, iterazioni):
    p =[]
    for k in range(iterazioni -3):
        p.append(np.log(abs(xk[k+2] - xk[k+3])/ abs(xk[k+1] - xk[k+2])) / np.log(abs(xk[k+1]-xk[k+2])/abs(xk[k]-xk[k+1])))
    ordine = p[-1]
    return ordine 

stima_ordine(xk, it)