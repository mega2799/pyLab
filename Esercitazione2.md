# ES 2

## 1

Si consideri il sistema di numeri macchina F(10, 2, −3, 3). Calcolare il punto medio del segmento [a, b] = [0.96e − 1, 0.99e − 1] secondo le formule _(a + b)/2_ e _a + (b − a)/2_

```py
import sympy as sym

import numpy as np

#F(10, 2, -3, 3) 

a = 0.96e-1

b = 0.99e-1

#sympy.Float(<operazione>, 2) arrotonda il risultato a 2 cifre siginificative

somma = sym.Float(a + b, 2)

pMedio = sym.Float(somma * 0.5, 2)

print(f'(a + b)/2 = {pMedio}')

differenza = sym.Float(b - a, 2)

somma = sym.Float(a + differenza,2) 

pMedio = sym.Float(somma + 0.5, 2)

print(f'(a + (b -a))/2 = {pMedio}')
```

## 3

Soluzione di un sistema lineare

```py
import numpy as np

import numpy.linalg as npl

import scipy.linalg  as spl

import matplotlib.pyplot as plt

A = np.array([ [3, 5], [3.01, 5.01] ]) # matrice dei coefficenti

b = np.array([10, 1]) # vettore dei risultati

x = spl.solve(A, b) # Soluzione del sistema

print(f'Vettore soluzione sistema: {x} %')


deltaA = np.array([ [0, 0], [0.01, 0]])

erroreDati = npl.norm(deltaA, np.inf) / npl.norm(A, np.inf) # dati perturbati/
                                                            # dati esatti

print(f'Errore relativo sui dati: {erroreDati * 100} %')

x1 = spl.solve(A + deltaA, b) # soluzione del sistema perturbato

print(f'Vettore soluzione sistema perturbato: {x1}')
```

## 4

Sistema lineare _Ax = b_, trovare vettore soluzione, perturbare la matrice dei coefficenti e calcolare errore relativo su soluzione e confrontarlo con la perturbazione su dati ingresso 

```py
import numpy as np

import numpy.linalg as npl

import scipy.linalg  as spl

import matplotlib.pyplot as plt

A = np.array([[6, 63, 662.2],[63, 662.2, 6967.8],[662.2, 6967.8, 73393.5664]])

b = np.array([1.1, 2.33, 1.7])

numeroDiCondizionamento = npl.cond(A, np.inf)

x = spl.solve(A,b)

print(f'Vettore soluzione del sistema: {x}')
    
print(f'{numeroDiCondizionamento=}')

Aperturbata = A.copy()

Aperturbata[0, 0] = A[0, 0] + 0.01

print(f'{Aperturbata=}')

x_perturbato = spl.solve(Aperturbata, b)

print(f'Vettore soluzione del sistema perturbato: {x_perturbato}')

erroreRelativoDati = npl.norm(A - Aperturbata, np.inf) / npl.norm(A, np.inf)

print("Errore relativo sui dati  in percentuale ", erroreRelativoDati *100,"%")

erroreRelativoSoluzione = npl.norm(x_perturbato - x, np.inf) / npl.norm(x, np.inf)

print("Errore relativo sulla soluzione  in percentuale ", erroreRelativoSoluzione *100,"%")
```

## 5

Assegnato il sistema lineare Ax = b, con A matrice di Hilbert di ordine 4 e b = [1, 1, 1, 1]^T
trovare il vettore soluzione x, perturbare il vettore dei termini noti, calcolare l’errore relativo sulla soluzione e confrontarlo con la perturbazione relativa sui dati di ingresso

```py
import numpy as np

import scipy.linalg as spl

import numpy.linalg as npl

n = 4 

A = spl.hilbert(n) # Matrice di Hilbert ordine 4

b = np.array([1, 1, 1, 1])

x = spl.solve(A, b)

print(f'Vettore Soluzione: {x}')

perturbazione = np.array([1, -1, 1, -1])

deltaB = b.copy()

deltaB = deltaB * perturbazione * 0.01

print(deltaB)

xPerturbato = spl.solve(A, b + deltaB)

print(f'Vettore Soluzione perturbato: {xPerturbato}')

err_dati = npl.norm(deltaB,np.inf)/npl.norm(b,np.inf)  

print("Errore relativo sui dati  in percentuale", err_dati*100,"%")

err_rel_sol = npl.norm(x-xPerturbato,np.inf)/npl.norm(x,np.inf) 

print("Errore relativo sulla soluzione  in percentuale", err_rel_sol*100,"%")
```

## 6 

Determinare l’intervallo I dei valori di a, b per cui l’algoritmo _fl(fl(a + b) ∗ fl(a − b))_ risulta numericamente piu' stabile di _fl(fl(a**2)-fl(b**2))

```py
import numpy as np

import scipy.linalg as spl

import numpy.linalg as npl

import sympy

# F(10, 5, L, U)

def func1(a, b):
    # (a - b) * (a + b)
    AmenoB = sympy.Float(a + b, 5)
    ApiuB = sympy.Float(a - b, 5)
    return sympy.Float(AmenoB * ApiuB, 5)

def func2(m, n):
    #(a^2 - b^2)
    A = sympy.Float(m**2, 1)
    B = sympy.Float(n**2, 5)
    print(A)
    print(B)
    return sympy.Float(1.0 - 1.9952, 5)

i = 0.1e1

j = 0.14125e1

scomposto = func1(i, j)

quadrati = func2(i, j)

risultato = i**2 - j**2

print(scomposto, quadrati, risultato)

errRelativoQuadrati = abs(-0.99520 - risultato) / abs(risultato)

errRelativoScomposto = abs(-0.99516 - risultato) / abs(risultato) 

print(f'{errRelativoQuadrati}')

print(f'{errRelativoScomposto}')
```
## 7

![](/img/2_7.png)

```py
"""
Calcolo exp(x) con serie di Taylor troncata in accordo a test
dell'incremento con x campionato in [a,b]
"""

import funzioniExp 

import numpy as np

import matplotlib.pyplot as plt

a = -10

b = 10

numCampioni = 10000

numList = np.linspace(a, b, numCampioni) # Returns num evenly spaced samples, calculated over the interval [start, stop].

expEsatta = np.exp(numList)

expApprossimata = np.zeros((numCampioni,)) # Return a new array of given shape and type, filled with zeros.

nIndexes = np.zeros((numCampioni,))

for i in range(numCampioni):
    expApprossimata[i], nIndexes[i]  = funzioniExp.esp_taylor_1(numList[i])

erroreRelativo = np.abs(expApprossimata - expEsatta) / np.abs(expEsatta)

plt.plot(numList, expApprossimata, '-b', numList, expEsatta, 'r--')

plt.title('Approssimazione esponenziale focon serie di Taylor troncata')

plt.legend(['expApprossimata','expEsatta'])

plt.show()


""" Coseh """


plt.plot(numList, erroreRelativo)

plt.title("Errore relativo scala cartesiana")

plt.show()


plt.plot(numList, erroreRelativo)

plt.yscale("log")

plt.title("Errore relativo scala semi-logaritimica")

plt.show()



plt.plot(numList, nIndexes)

plt.title('Indice n')

plt.show()

"""
--------------------------------------------------------------------------
come migliorare andamento errore relativo
--------------------------------------------------------------------------
"""

for i in range(numCampioni) :
    if numList[i]>=0:
        expApprossimata[i], nIndexes[i] = funzioniExp.esp_taylor_1(numList[i])
    else:
        expApprossimata[i], nIndexes[i] = funzioniExp.esp_taylor_2(numList[i])


err_rel_2 = np.abs(expApprossimata - expEsatta)/np.abs(expEsatta)

plt.plot(numList, err_rel_2)

plt.yscale("log")

plt.title('Errore relativo Algoritmo Migliorato - scala semilogaritmica')

plt.show()
```

## 8 

Calcolare l’approssimazione della derivata prima di f(x) = sin(x) in x = 1 mediante il rapporto incrementale (f(x + h) − f(x))/h per valori decrescenti di h, confrontandolo con il valore fornito dalla funzione di libreria per f'(x) mediante calcolo dell’errore relativo

```py
import numpy as np

import math

import matplotlib.pyplot as plt

k = np.arange(0, -21, -1) #Return evenly spaced values within a given interval, Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop).
# array di numeri negativi [0,-1 .....-20]
h=10.0**k
#array di numeri piccoli
derivataEsatta = math.cos(1)

x = 1

rapportoIncrementale = (np.sin(x + h) - np.sin(x)) / h

erroreRelativo = np.abs(rapportoIncrementale - derivataEsatta) / np.abs(derivataEsatta)


plt.plot(h, erroreRelativo, 'b-', h, h, 'r:')

plt.xscale("log")

plt.yscale("log")

plt.legend(['Errore relativo', 'Incremento'])
```
