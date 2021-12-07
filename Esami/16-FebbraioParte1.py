import matplotlib
import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

# Def funzione 

x = sym.symbols('x')

fx = x - 1/3 * sym.sqrt(30 * x -25)

f = lambdify(x, fx, np) 

a = 5/6 

b = 25/6 

# a) graficamente noto che ha 1 soluzione

xx = np.linspace(a, b, 200)

plt.plot(xx, f(xx))
plt.axhline(y=0, color='r', ls='--')
plt.axvline(x=0,color='r', ls='--' )
plt.show()

# b) con un metodo iterativo che converge quadraticamente si intende newton

x0 = 4 

toll = 1e-12 

dfx = sym.diff(fx, x, 1)

df = lambdify(x, dfx, np)

#Newton Modificato
def newton_m(fname,fpname,x0,m,tolx,tolf,nmax):
        eps=np.spacing(1)     
        xk=[]
        #xk.append(x0)
        fx0=fname(x0)
        dfx0=fpname(x0)
        if abs(dfx0)>eps:
            d=fx0/dfx0
            x1=x0-m*d
            fx1=fname(x1)
            xk.append(x1)
            it=0
           
        else:
            print('Newton:  Derivata nulla in x0  \n')
            return [],0,[]
        
        it=1
        while it<nmax and abs(fx1)>=tolf and  abs(d)>=tolx*abs(x1):
            x0=x1
            fx0=fname(x0)
            dfx0=fpname(x0)
            if abs(dfx0)>eps:
                d=fx0/dfx0
                x1=x0-m*d
                fx1=fname(x1)
                xk.append(x1)
                it=it+1
            else:
                 print('Newton Mod: Derivata nulla   \n')
                 return x1,it,xk           
           
        if it==nmax:
            print('Newton Mod: raggiunto massimo numero di iterazioni \n');
        
        return x1,it,xk

x1, iterazioni, soluzioni = newton_m(f, df, x0, 2, toll, toll, 100) # quel 2 indica la molteplicita' degli zeri, ossia interseca l'asse delle x una volta sola ma essendo di grado due ha doppia molteplicita

print("soluzione: " + str(x1))

# c) ordine di convergenza con funzione stima dell ordine

def stima_ordine(xk,iterazioni):
      p=[]

      for k in range(iterazioni-3):
         p.append(np.log(abs(xk[k+2]-xk[k+3])/abs(xk[k+1]-xk[k+2]))/np.log(abs(xk[k+1]-xk[k+2])/abs(xk[k]-xk[k+1])));
     
      ordine=p[-1]
      return ordine

print(stima_ordine(soluzioni, iterazioni))

# d) 
plt.semilogy(np.abs(soluzioni), range(iterazioni))
plt.show()

# e) come nel punto b) ma con x0 = 5/6 

x0 = 5/6 

x1, iterazioni, soluzioni = newton_m(f, df, x0, 2, toll, toll, 100) # x1 = 0.8333333333333334 

# non converge ad alpha 