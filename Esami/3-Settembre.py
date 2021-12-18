import numpy as np

import sympy as sym

import matplotlib.pyplot as  plt 

import numpy.linalg as npl 

import math

import sympy.utilities.lambdify as lambdify

def plagr(xnodi, k):
    xzeri = np.zeros_like(xnodi)
    n = xnodi.size
    if k == 0:
        xzeri = xnodi[1:n]
    else:
        xzeri = np.append(xnodi[0:k], xnodi[k+1:n])
    num = np.poly(xzeri)
    den = np.polyval(num, xnodi[k])
    p = num / den 
    return p

def interpl(x, f, xx):
    n = x.size 
    m = xx.size 
    L = np.zeros((n,m))
    for k in range(n):
        p = plagr(x, k)
        L[k, :] = np.polyval(p, xx)
    return np.dot(f, L)


# def 

a = 0 

b = 2 

x = sym.symbols('x')

fx = (1 / (2 + sym.sin(sym.pi * x))) - x**2 * sym.cos(sym.pi * x)

# c)
f = lambdify(x, fx, np)

c = np.array([0.5, 1, 1.5])

y = f(c)

xx = np.linspace(a, b, 100)

pol = interpl(c, y, xx)
# e) 
plt.plot(xx, pol, xx, f(xx), c, y, 'o', xx , abs(f(xx) - pol))
plt.legend(['pol interpolante', 'funzione f(x)', 'nodi', 'funzione resto'])
plt.axvline(x=0)
plt.axhline(y=0)
plt.show()

# f)

index = 0
max = abs(f(xx)[0] - pol[0])
for i in range(len(xx)):
    if abs(f(xx[i]) - pol[i]) > max:
        max = abs(f(xx)[i] - pol[i])
        index = i
print(max, i)