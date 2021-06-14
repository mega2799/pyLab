import numpy as np

import sympy as sym 

a = 0

b = 1

toll = 1e-6

def trapcomp(fname, a, b, n):
    h = (b-a)/n
    nodi = np.arange(a, b+h, h)
    f = fname(nodi)
    I = (f[0] + 2*np.sum(f[1:n] + f[n]) )* h/2
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

res = []
for i in range(1,31):
    f = lambda x: x**i/(x+10)
    res.append(trapToll(f, a, b, toll)[0])

print(res)

ipsilon = [np.log(11) - np.log(10)]
for i in range(1, 30):
    ipsilon.append(1/i - 10*ipsilon[i-1])

print(ipsilon)

