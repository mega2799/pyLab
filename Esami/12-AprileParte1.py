import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

# def funzione 

a = 0 

b = 1

n = 30 

x = sym.symbols('x')

fx = lambda i : x**i / (x + 10)

toll = 1e-6 

def TrapComp(fname,a,b,n): # Trapezi composti
    h = (b-a)/n # Passo
    nodi = np.arange(a, b+h, h)
    f = fname(nodi)
    I = (f[0] + 2*np.sum(f[1:n]) + f[n]) * h/2
    return I
    
def traptoll(fun,a,b,tol):
    # Ripete la formula dei trapezi composti ma stavolta ho una
    # precisione (tolleranza) 
    Nmax=2048
    err=1
    
    N=1
    IN=TrapComp(fun,a,b,N)
    
    while N<=Nmax and err>tol :
        N = 2*N # Raddoppio il numero dei sottointervalli
        I2N = TrapComp(fun,a,b,N)
        err = abs(IN-I2N)/3
        IN = I2N
 
    
    if N>Nmax:
        print('Raggiunto nmax di intervalli con traptoll')
        N=0
        IN=[]
 
    return IN,N

vettoreSoluzioni = []
for i in range(1, n+1):
    f = fx(i)
    f = lambdify(x, f, np)
    vettoreSoluzioni.append(traptoll(f, a, b, toll)[0])

# b) con algoritmo 

y1 = np.zeros((n,), dtype=float)

y1[0] = math.log(11) - math.log(10)

for i in range(1, 30):
    y1[i] = 1/i - 10 * y1[i - 1]

# c) con algoritmo 

z1 = np.zeros((n +1 ,), dtype=float)

for i in range(30, 0, -1):
    z1[i - 1] = 1/10 * (1/i + z1[i])

# d) 

ery = np.abs(y1 - vettoreSoluzioni) / np.abs(y1)

erz = np.abs(z1[:-1] - vettoreSoluzioni) / np.abs(z1[:-1]) # vettore z1 era di 31 el, quindi elimino l'ultimo (che vale 0) con [:-1]

# e) 

plt.plot(range(n), ery, range(n), erz)
plt.legend(['funzione y', 'funzione z'])
plt.show()

# L'algoritmo piu stabile sembrerebbe quello della z, ho uno 
# spike improvviso che si discosta dai valori esatti in y 
