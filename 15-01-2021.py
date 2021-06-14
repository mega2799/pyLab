import numpy as np

import matplotlib.pyplot as plt 

import sympy as sym

from sympy.utilities.lambdify import lambdify

f = lambda x: (15*((3/5)**x + 1) / (5*(3/5)**x +3))

f1 = lambda x: 8 - (15/x)

f2 = lambda x,y: 108 - (815/x) + 1500/(x*y)

n = 35

v = np.zeros((n,), dtype=float)
for i in range(0, n-1):
    v[i] = f(i) 

b = np.zeros((n,), dtype=float)
b[0] = (4.0)
for i in range(1,n-1):
    b[i]=f1(b[i-1])

c = np.zeros((n,), dtype=float)
c[0] = 4.0
c[1] = 17/4
for i in range(2, len(v)):
    c[i] = f2(c[i-1], c[i-2])

print(v)

print(b) # Converge a 5 

print(c) # Converge a 100

# Grafico per gli errori 

# Formula (2)
errRelativo2 = np.abs(b - v)/np.abs(v) 
#errRelativo2 = [abs(v[x] - b[x])/abs(v[x]) for x in range(len(v))]
#Formula (3)
errRelativo3 = np.abs(c - v)/np.abs(v) 
#errRelativo3 = [abs(v[x] - c[x])/abs(v[x]) for x in range(len(v))]

plt.semilogy(range(n), errRelativo2, range(n), errRelativo3)
plt.legend(['Errore relativo formula 2', 'Errore relativo formula 3'])
plt.show()

# Punto fisso 
x = sym.symbols('x')
xx = np.linspace(4, 100, 100) # dati dal problema 

#f1 
f1x = 8-15/x 
derivataf1x = sym.diff(f1x, x, 1)
derivataf1 = lambdify(x, derivataf1x, np)
f1 = lambdify(x, f1x, np)

plt.plot(xx, f1(xx)) # La funzione su un intervallo
plt.plot(xx, xx) # y=x 
plt.legend(["f1(x)", "y=x"])
plt.show()

#f2
f2x = 108 - (815/x) + 1500/(x*x)
derivataf2x = sym.diff(f2x, x, 1)
derivataf2 = lambdify(x, derivataf2x, np)
f2 = lambdify(x, f2x, np)

plt.plot(xx, f2(xx))
plt.plot(xx, xx)
plt.legend(["f2(x)", "y=x"])
plt.show()

def iterazione(fname, x0, tolx, nmax):
    xk =[]
    xk.append(x0)
    x1 = fname(x0)
    d = x1 - x0 
    xk.append(x1)
    it = 1
    while it < nmax and abs(d) >= tolx * abs(x1):
        x0 = x1 
        x1 = fname(x0)
        d = x1 - x0 
        xk.append(x1)
        it += 1

        if(it == nmax):
            print("raggiunto max")
    return x1, it, xk 

tolx = 1e-5
nmax = 100
x0 = 4 
x1, it, xk = iterazione(f1, x0, tolx, nmax)
print("punto fisso di f1: ", x1) 

x1, it, xk = iterazione(f2, x0, tolx, nmax)
print("punto fisso di f2: ", x1) 

#Verifico che 5 sia punto fisso di f1 
zz = np.linspace(0, 6,100) # (0,6) intervallo in cui 5 è compreso e si vede....
plt.semilogy(zz, derivataf1(zz), 5, derivataf1(5), 'o')
plt.plot([0,6], [1, 1], '--')
plt.plot([0,6], [-1, -1], '--')
plt.legend(['derivata prima di g1 in un intorno di 5 ','punto fisso' ,'y=1','y=-1'])
plt.show()

#Verifico se 5 sia punto fisso di f2 
zz = np.linspace(0, 6,100) # (0,6) intervallo in cui 5 è compreso e si vede....
plt.semilogy(zz, derivataf2(zz), 5, derivataf2(5), 'o')
plt.plot([0,6], [1, 1], '--')
plt.plot([0,6], [-1, -1], '--')
plt.legend(['derivata prima di g2 in un intorno di 5 ','punto fisso' ,'y=1','y=-1'])
plt.show()
# Il punto fisso qua non è compreso => allora 5 non e punto fisso 

#Verifico se 100 sia punto fisso di f2 
zz = np.linspace(97, 101,100) # (0,6) intervallo in cui 5 è compreso e si vede....
plt.semilogy(zz, derivataf2(zz), 100, derivataf2(100), 'o')
plt.plot([97,101], [1, 1], '--')
plt.plot([97, 101], [-1, -1], '--')
plt.legend(['derivata prima di g2 in un intorno di 100 ','punto fisso' ,'y=1','y=-1'])
plt.show()

