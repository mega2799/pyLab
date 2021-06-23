import numpy as np 

import sympy as sym

import matplotlib.pyplot as plt 

from sympy.utilities import lambdify

from scipy.optimize import fsolve

a = 5/6 

b = 25/6

f = lambda x: x - (1/3)*np.sqrt(30*x - 25)

# Quante radici reali nell intervallo ? 

xx = np.linspace(a, b, 200) 
plt.plot(xx, f(xx))
plt.legend(['f(x) in [a,b]'])
plt.show()

#Sembra avere 1 radice
x = sym.symbols('x')
x0 = 4
fx = x - (1/3)* sym.sqrt(30*x - 25)
fx1 = lambdify(x, fx, np)
alpha = fsolve(fx1, x0)
print(f'{alpha=}')

df = sym.diff(fx, x, 1)
dfx = lambdify(x, df, np)

print("la derivata prima in alpha vale ", dfx(alpha))

# Metodo iterativo che con x0 = 4 converga quadraticamente a alpha
def newton(fname, fpname, x0, a, b, tollf, tollx, iterazione):
    xk = []
    fx0 = fname(x0)
    dfx0 = fpname(x0)
    if abs(dfx0) > np.spacing(1):
        it = 0
        d = fx0 / dfx0
        x1 = x0 - d 
        fx1 = fname(x1)
        xk.append(x1)
    else:
        print("derivata nulla") 
        return 0, 0, []
    while it < iterazione and np.abs(fx1) >= tollf and np.abs(d) >= tollx * np.abs(x1):
        x0 = x1 
        fx0 = fname(x0)
        dfx0 = fpname(x0)
        if abs(dfx0) > np.spacing(1):
            it += 1
            d = fx0 / dfx0
            x1 = x0 - d
            fx1 = fname(x1)
            xk.append(x1)
        else:
            print("derivata nulla") 
            return 0, 0, []
    if it == iterazione:
        print("raggiunto nmax")

    return x1, it, xk

x1, it, kk = newton(fx1, dfx, x0, a, b, 1e-12, 1e-12, 200)

print(x1) 
# Sono uguali 

# Confermo l'ordine di convergenza

def iterazione(xk, it):
    p = []
    for k in range(it -3):
        num = np.abs((xk[k+2] - xk[k+3]) / xk[k+1] - xk[k + 2]) 
        den = np.abs((xk[k+1] - xk[k+2]) / xk[k] - xk[k + 1]) 
        p.append(num/den)
    return p[-1]

stima = iterazione(kk, it)

print("la stima vale", stima)

