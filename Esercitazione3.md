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
