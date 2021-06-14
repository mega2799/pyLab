import numpy as np

import sympy as sym 

import matplotlib.pyplot as plt
a = 0

b = 1

toll = 1e-6

def trapcomp(fname, a, b, n):
    h = (b-a)/n
    nodi = np.arange(a, b+h, h)
    f = fname(nodi)
    I = (f[0] + 2*np.sum(f[1:n])+ f[n] )* h/2
    return I

def trapToll(fname, a, b, toll):
    Nmax = 2048 
    err = 1

    N = 1
    IT = trapcomp(fname, a, b, N)

    while err > toll and N <= Nmax:
        N = 2 * N 
        I2N = trapcomp(fname, a, b, N)
        err = abs(IT - I2N) / 3
        IT = I2N

    if N > Nmax:
        print("max raggiunto") 
        return [], 0

    return IT, N

# Punto a
res = []
for i in range(1,31):
    f = lambda x: x**i/(x+10)
    res.append(trapToll(f, a, b, toll)[0])

print(res)

# Punto b
ipsilon = np.zeros((30,), dtype=float)
ipsilon[0] = np.log(11) - np.log(10)
for i in range(1, 30):
    ipsilon[i] = (1/i - 10*ipsilon[i-1])

print(ipsilon)

# Punto c
zeta = np.zeros((31,),dtype=float)

for i in range(30, 0, -1):
    zeta[i-1] = .1*(1/i - zeta[i])
print(zeta)

errAlgoritmoB = np.abs(res - ipsilon) / np.abs(res)

errAlgoritmoC = np.abs(res - zeta[0:30]) / np.abs(res)

plt.semilogy(np.arange(30), errAlgoritmoB, 'g-.', np.arange(30), errAlgoritmoC, 'b--')
plt.legend(['Errore relativo algoritmo b ', 'Errore relativo algoritmo c'])
plt.show()

