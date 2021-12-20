import numpy as np

import sympy as sym 

import matplotlib.pyplot as plt 

from sympy.utilities import lambdify

import scipy.linalg as spl

import math 


# def 

x = sym.symbols('x')

fx = sym.sin(x)  + sym.sin(6*x)

f = lambdify(x, fx, np)

a = - np.pi / 2 

b = 5*np.pi / 2 

x1 = np.arange(0, 2*np.pi, 2/7*np.pi)

y1 = f(x1)

x2 = np.array([np.pi/7, np.pi/3, 2*np.pi/3, 4*np.pi/3, 5*np.pi/3, 13*np.pi/7], dtype=float)

y2 = f(x2)


# a) 
def Usolve(U,b):
    m,n = U.shape
    flag = 0
    if n!= m :
        print('non quadr')
        flag = 1 
        x = []
        return x,  flag 
    if np.all(np.diag(U)) != True:
        print('diag nulla')
        x = [] 
        flag = 1
        return x, flag 
    
    x = np.zeros((n,1))
    for i in range(n-1, -1, -1):
        s = np.dot(U[i, i+1:n], x[i+1:n])
        x[i] = (b[i] - s)/U[i,i]
    return x, flag 

def metodoQR(x, y, n):
    H = np.vander(x, n+1)
    Q, R = spl.qr(H)
    y1 = np.dot(Q.T, y)
    a, flag = Usolve(R[0:n+1, :], y1[0:n+1])
    return a 

# b) 

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
    

xx = np.linspace(a, b, 200)

grado = 3 

# c1) 

pol1 = metodoQR(x1, y1, grado)

pol2 = metodoQR(x2, y2, grado)

p1 = np.polyval(pol1, xx) 

p2 = np.polyval(pol2, xx) 


plt.plot(xx, f(xx), x1, y1, x2, y2, xx, p1, xx, p2)
plt.legend(['funzione f(x)','funzione f1(x)', 'funzxione f2(x)', 'polinomio qr 1', 'polinomio qr 2'])
plt.show()

# c2)

pol1 = interpl(x1, y1, xx)

pol2 = interpl(x2, y2, xx)

plt.plot(xx, f(xx), x1, y1, x2, y2, xx, pol1, xx, pol2)
plt.legend(['funzione f(x)','funzione f1(x)', 'funzxione f2(x)','polinomio interpolante 1', 'polinomio interpolante 2'])
plt.show()