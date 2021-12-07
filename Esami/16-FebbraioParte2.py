import matplotlib
import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

f= lambda x: np.cos(math.pi*x)+np.sin(math.pi*x)

a = 0 
 
b = 2 

def plagr(xnodi,k):
    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if k==0:
       xzeri=xnodi[1:n]
    else:
       xzeri=np.append(xnodi[0:k],xnodi[k+1:n])
    
    num=np.poly(xzeri) 
    den=np.polyval(num,xnodi[k])
    
    p=num/den
    
    return p

def InterpL(x, f, xx):
     n=x.size
     m=xx.size
     L=np.zeros((n,m))
     for k in range(n):
        p=plagr(x,k)
        L[k,:]=np.polyval(p,xx)
    
    
     return np.dot(f,L)
# c) 

nodi = np.array([1.0, 1.5, 1.75]) # nodi del problema

xx = np.linspace(a, b, 200)

y = f(nodi) # con la y trovo i punti in cui interpolare la funzione

pol = (InterpL(nodi, y, xx)) # funzione.....

# d) 

plt.plot(xx,pol,'r-',xx,f(xx),'b--',nodi,y,'go')
plt.legend(['interpolante di Lagrange','Funzione da interpolare', 'nodi di interpolazione'])
plt.show() 

# e) 


px = InterpL(nodi, y, np.array([.75])) # equivale a.... pol(1.75)  

resto = np.abs(px - f(.75)) 

# f) 

nodi = np.array([.75 ,1.0, 1.5, 1.75]) # nodi del problema putno f

pol = (InterpL(nodi, y, xx)) 