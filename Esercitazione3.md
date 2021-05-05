# Esercitazione 3

## 1
```py
import numpy as np

import sympy as sym 

import matplotlib.pyplot as plt 

from sympy.utilities.lambdify import lambdify

import funzioniZeri

scelta = input("Scegli quale tra le 3 funzioni\n")

x = sym.symbols('x')

# key : [ funzioneSimbolica, f(0), estremoDX, estremoSX, valore innesco x0, , secondoIterato(secanti) ] 

functions = {
        '1': [sym.exp(-x)-(x+1),0,-1,2,-0.5,-0.3],
        '2': [sym.log(x+3,2)-2,1,-1,2,-0.5,0.5],
        '3': [sym.sqrt(x)-x**2/4, 2**(4/3),1,3,1.8,1.5]
    }

func = functions.get(scelta)

f, alpha, a, b, x0, xm1 = func

deltaF = sym.diff(f, x, 1)

#Rendo numeriche la f e la derivata

fNumerica = lambdify(x, f, np)

deltaFNumerica = lambdify(x, deltaF, np)

insiemeNum = np.linspace(a, b, 100)

plt.plot(insiemeNum, 0 * insiemeNum, fNumerica(insiemeNum), 'r-')

funcName = 'Function ' + str(f)
plt.title(funcName)

plt.show()

tolleranzax = 1e-12

#Applico il metodo di bisezione
xbis,itbis,xkbis = funzioniZeri.bisez(fNumerica, a, b, tolleranzax)

tolleranzaF=1e-12
nmax=500
#Applico il metodo delle corde
xcorde, itcorde, xkcorde = funzioniZeri.corde(fNumerica, deltaFNumerica, x0, tolleranzax, tolleranzaF, nmax)

#Applico il metodo della regula falsi
xfalsi, itfalsi, xkfalsi = funzioniZeri.regula_falsi(fNumerica, a, b, tolleranzax, nmax)

#Applico il metodo di Newton
xNew, itNew, xkNew = funzioniZeri.newton(fNumerica, deltaFNumerica, x0, tolleranzax, tolleranzaF, nmax)

#Applico il metodo delle secanti
xSec, itSec, xkSec = funzioniZeri.secanti(fNumerica, xm1, x0, tolleranzax,tolleranzaF,nmax)

# Calcolo degli errori relativi 

err_bis = np.abs(np.array(xkbis) - alpha)

err_corde = np.abs(np.array(xkcorde) - alpha)

err_falsi = np.abs(np.array(xkfalsi) - alpha)

err_New = np.abs(np.array(xkNew) - alpha)

err_sec = np.abs(np.array(xkSec) - alpha)


plt.semilogy(range(itbis), err_bis, 'go-', range(itfalsi), err_falsi, 'mo-', range(itcorde), err_corde, 'bo-', range(itSec), err_sec, 'co-', range(itNew), err_New,'ro-')

plt.legend([ 'Bisezione', 'Regula Falsi', 'Corde', 'Secanti', 'Newton'])

plt.title('grafico errori metodi itertivi rispetto risultato esatto f(0) su numeri di iterazioni')

plt.show()

#Calcolo l'ordine di convergenza di ogni metodo, richiamando la funzione stima_ordine del modulo funzione_zeri
ordine_bis = funzioniZeri.stima_ordine(xkbis,itbis)

ordine_corde = funzioniZeri.stima_ordine(xkcorde,itcorde)

ordine_falsi = funzioniZeri.stima_ordine(xkfalsi,itfalsi)

ordine_New  =funzioniZeri.stima_ordine(xkNew,itNew)

ordine_sec = funzioniZeri.stima_ordine(xkSec,itSec)


print("Bisezione it={:d}, ordine di convergenza {:e}".format(itbis,ordine_bis))

print("Corde it={:d}, ordine di convergenza {:e}".format(itcorde,ordine_corde))

print("Falsi it={:d}, ordine di convergenza {:e}".format(itfalsi,ordine_falsi))

print("Newton it={:d}, ordine di convergenza {:e}".format(itNew,ordine_New))

print("Secanti it={:d}, ordine di convergenza {:e}".format(itSec,ordine_sec))
```

## 2

```py
import math

import numpy as np

import sympy as sym

import funzioniZeri

import matplotlib.pyplot as plt

from sympy.utilities.lambdify import lambdify

tolleranzaX = 1e-8

tolleranzaF = 1e-8

a = 3/5*math.pi

b = 37/25*math.pi

x = sym.symbols('x') 

fx = sym.tan(x) - x

deltaF = sym.diff(fx, x, 1) 

#Trasformo in numeriche la funzione e la sua derivata

f = lambdify(x, fx, np)

df = lambdify(x, deltaF, np)

insiemeNum = np.linspace(a, b, 100)

plt.plot(insiemeNum, f(insiemeNum), 'r-')

plt.plot(insiemeNum, 0 * insiemeNum, insiemeNum, f(insiemeNum), 'r-') #asse X 

plt.show()

#Metodo di bisezione

xbis, itbis, xkbis = funzioniZeri.bisez(f, a, b, tolleranzaX)

print('zero bisezione= {:e} con {:d} iterazioni \n'.format(xbis, itbis))

nmax = 200

vettx0 = xkbis[0:4]

for j in range(4):
    x0 = xkbis[j]

    xNew,itNew,xkNew = funzioniZeri.newton(f, df, x0, tolleranzaX, tolleranzaF, nmax)
    
    print('------------------------------' + 'caso J: ' + str(j) + '------------------------------------------------')
    
    print('X0= {:e} ,  zero Newton= {:e} con {:d} iterazioni \n'.format(x0, xNew, itNew))
    
    xm1=a
        
    xSec, itSec, xkSec= funzioniZeri.secanti(f, xm1, x0, tolleranzaX, tolleranzaF, nmax)
    
    print('X0= {:e}  zero Secanti= {:e} con {:d} iterazioni \n'.format(x0, xSec, itSec))

    print('------------------------------------------------------------------------------')

"""
%per j=0,1: 
% -Newton non converge
% -secanti non converge alla radice richiesta, ma a 0.002
%per j=2:
% -Newton converge alla radice richesta
% -secanti non converge alla radice richiesta, ma a 0.002
%per j=3:
% entrambi convergono alla radice richiesta
""" 
```

## 3

```py
import math

import numpy as np

import sympy as sym

import funzioniZeri

import matplotlib.pyplot as plt

from sympy.utilities.lambdify import lambdify

tolleranzaX = 1e-6

tolleranzaF = 1e-5

x = sym.symbols('x') 

fx = sym.atan(x) 

deltaF = sym.diff(fx, x, 1) 

#Trasformo in numeriche la funzione e la sua derivata

f = lambdify(x, fx, np)

df = lambdify(x, deltaF, np)

insiemeNum = np.linspace(-10, 10, 100)

plt.plot(insiemeNum, 0 * insiemeNum, insiemeNum, f(insiemeNum), 'r-') #asse X 

plt.show()

nmax = 500


#Considero come iterato iniziale per Newton: x0=1.2: il metodo converge
x0=1.2

xNew, itNew, xkNew=funzioniZeri.newton(f, df, x0, tolleranzaX, tolleranzaF, nmax)

print('X0= {:e} ,  zero Newton= {:e} con {:d} iterazioni \n'.format(x0, xNew, itNew))

#Considero come iterato iniziale per Newton: x0=1.2: il metodo non converge  
x0=1.4

xNew, itNew, xkNew=funzioniZeri.newton(f, df, x0, tolleranzaX, tolleranzaF, nmax)

print('X0= {:e} ,  zero Newton= {:e} con {:d} iterazioni \n'.format(x0, xNew, itNew))
```

## 4 
```py
import math

import numpy as np

import sympy as sym

import funzioniZeri

import matplotlib.pyplot as plt

from sympy.utilities.lambdify import lambdify

tolleranzaX = 1.e-12

tolleranzaF = 1.e-12

x = sym.symbols('x')

fname = f = x**3 + x**2 -33*x + 63

deltaF = sym.diff(f, x, 1)

#Trasformo in numeriche le funzioni 

f = lambdify(x, f, np)

deltaF = lambdify(x, deltaF, np)

insiemeNum = np.linspace(-10, 10, 100)

plt.plot(insiemeNum, 0 * insiemeNum, insiemeNum, f(insiemeNum), 'r-')

plt.title(fname)

plt.show() #grafico della funzione 

nmax = 500

x0 = 1

xNew, itNew, xkNew = funzioniZeri.newton(f, deltaF, x0, tolleranzaX, tolleranzaF, nmax)

print('X0= {:e} ,  zero Newton= {:e} con {:d} iterazioni \n'.format(x0, xNew, itNew))

ordine_New = funzioniZeri.stima_ordine(xkNew, itNew)

print("Newton it={:d},  ordine di convergenza {:e}".format(itNew, ordine_New))

#Utilizzando il metodo di Newton modificato e ponendo m uguale alla molteplicità della radice si ottiene un metodo con ordine di convergenza 2

m = 2

xNew_m, itNew_m, xkNew_m = funzioniZeri.newton_m(f, deltaF, x0, m, tolleranzaX, tolleranzaF, nmax)

print('X0= {:e} ,  zero Newton Mod= {:e} con {:d} iterazioni \n'.format(x0, xNew_m, itNew_m))

ordine_New_m = funzioniZeri.stima_ordine(xkNew_m, itNew_m)

print("Newton Mod it={:d},  ordine di convergenza {:e}".format(itNew_m, ordine_New_m))

```

## 5
```py
import numpy as np

import sympy as sym

import funzioniZeri

import matplotlib.pyplot as plt

from sympy.utilities.lambdify import lambdify

x0 = 2.5

tolleranzaX = 1.8e-8

nmax = 1000

#su virtuale spiega come ci è arrivata....
c=[1/20, 1/6, 3/10, 2/5]

x = sym.symbols('x')

for i in range(len(c)):
    gx = x -c[i]*(x**2 -5)
    deltaGx = sym.diff(gx, x, 1)
    g = lambdify(x, gx, np)
    deltaGx = lambdify(x, deltaGx, np)
    x1, it, xk = funzioniZeri.iterazione(g, x0, tolleranzaX, nmax)

    print('iterazioni= {:d}, soluzione={:e} \n\n'.format(it, x1))
    deltaGx1 = abs(deltaGx(x1)) # x1 della derivata 
    xx = np.linspace(1.5, 3)

    plt.plot(xx, xx, 'k-', xx, g(xx))
    plt.title("abs(g'(alfa))="+str(deltaGx1))

    #Grafico della poligonale  
    Vx=[]
    Vy=[]
    for k in range(it):
        Vx.append(xk[k])
        Vy.append(xk[k])
        Vx.append(xk[k])
        Vy.append(xk[k+1])
    
    Vy[0]=0
    plt.plot(Vx,Vy,'r',xk,[0]*(it+1),'or-')

  
    plt.show()
```

## 6

```py
import numpy as np

import sympy as sym

import funzioniZeri

import matplotlib.pyplot as plt

from sympy.utilities.lambdify import lambdify

tolleranzaX = 1.e-7

nmax = 1000

f = lambda x: x**3 + 4*x**2 - 10 

insiemeNum = np.linspace(0.0, 1.6, 100)

plt.plot(insiemeNum, 0*insiemeNum, insiemeNum, f(insiemeNum))

plt.title("asse x e la funzione f valutata in un intervallo opportuno")

plt.show()

x0 = 1.5

x = sym.symbols('x')

gx = sym.sqrt(10 / (x + 4))


g = lambdify(x, gx, np)

plt.plot(insiemeNum, insiemeNum, insiemeNum, g(insiemeNum))

plt.title('funzione g(x) e y=x')

plt.show()

deltaGx = sym.diff(gx, x, 1)

deltaGx = lambdify(x, gx, np)

plt.plot(insiemeNum, deltaGx(insiemeNum))

plt.title('funzione deltaGx')

plt.show()

x1, it, xk = funzioniZeri.iterazione(g, x0, tolleranzaX, nmax)

print('iterazioni= {:d}, soluzione = {:e} \n\n'.format(it, x1))

ordineIterazioni = funzioniZeri.stima_ordine(xk, it)
#Essendo il metodo con ordine di convergenza lineare, la costante asintotica di convergenza è data
#da |g'(alfa)| dove alfa è la radice

C = abs(deltaGx(x1))

print("Iterazione it={:d}, ordine di convergenza {:e}, Costante asintotica di convergenza {:e}".format(it,ordineIterazioni,C)) 

plt.plot(insiemeNum, insiemeNum, 'k-', insiemeNum, g(insiemeNum))

plt.title("abs(g'(alfa))="+str(C))

Vx=[]
Vy=[]
for k in range(it):
    Vx.append(xk[k])
    Vy.append(xk[k])
    Vx.append(xk[k])
    Vy.append(xk[k+1])
    
Vy[0]=0
plt.plot(Vx,Vy,'r',xk,[0]*(it+1),'or-')
plt.show()

# Si osserva che a parità di ordine di convergenza, più piccola è la costante asintotica di convergenza,
#maggiore è la velocità del metodo.
```
