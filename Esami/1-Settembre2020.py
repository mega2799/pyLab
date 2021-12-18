import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

import numpy.linalg as npl

from scipy.linalg import solve

# def 

x = sym.symbols('x')

fx = x - 2 * sym.sqrt(x - 1)

f = lambdify(x, fx, np)

def newton_m(fname, fpname, x0, m, tolx, tolf, nmax):
    eps = np.spacing(1)
    xk = []
    fx0 = fname(x0)
    dfx0 = fpname(x0)
    if abs(dfx0) > eps:
        d = fx0/dfx0
        x1 = x0 - m*d 
        fx1 = fname(x1)
        xk.append(x1)
        it=0
    else:
        print('derivata nulla')
        return [], 0, []
    it=1
    while it < nmax and abs(fx1) >= tolf and abs(d) >= tolx*abs(1):
        x0 = x1 
        fx0 = fname(x0)
        dfx0 = fpname(x0)
        if abs(dfx0) > eps:
            d = fx0/dfx0
            x1 = x0-m*d 
            fx1 = fname(x1)
            xk.append(x1)
            it += 1
        else:
            print('derivata nulla')
            return x1, it, xk
    if it == nmax:
        print('it max')
    return x1, it, xk 








def stima_ordine(xk,iterazioni):
      p=[]

      for k in range(iterazioni-3):
         p.append(np.log(abs(xk[k+2]-xk[k+3])/abs(xk[k+1]-xk[k+2]))/np.log(abs(xk[k+1]-xk[k+2])/abs(xk[k]-xk[k+1])));
     
      ordine=p[-1]
      return ordine


a = 1 

b = 3 

# a) 

xx = np.linspace(a, b, 100)

plt.plot(xx, f(xx))
plt.axvline(x=0)
plt.axhline(y=0)
plt.show()

# ha due radici reali, posso vederlo graficamente, che sembrano coincidere in una sola

# b) 

x0 = 3

deltafx = sym.diff(fx, x, 1)

df = lambdify(x, deltafx, np)

alpha, it, xk = newton_m(f, df, x0, 2, 1e-12, 1e-12, 100)

print(alpha, it, xk)

ordine = stima_ordine(xk, it)

print(ordine)