# Esercitazione 4

![testo](./src/testo_esercitazione4.pdf)

## 1

```py
import numpy as np

import time 

import math

import numpy.linalg as npl

import scipy.linalg as spl

import matplotlib.pyplot as plt

import Sistemi_lineari 

numInizio = 5

numFine = 100

tempoNoPivot = [] # Array dei tempi risoluzione con f Sistemi_lineari.LU_nopivot 

erroreRelativoNoPivot  = [] # Array errori risoluzione con f Sistemi_lineari.LU_nopivot 

tempoPivot = [] # Array dei tempi risoluzione con f Sistemi_lineari.LU_pivot 

erroreRelativoPivot  = [] # Array errori risoluzione con f Sistemi_lineari.LUpivot 

tempoLib = [] # Array dei tempi risoluzione con scipy

erroreRelativoLib  = [] # Array errori risoluzione con scipy


for n in range(numInizio, numFine):
    A = np.empty((n,n), dtype=float) # Matrice di tipo 1x1, 2x2, 3x3....
    for i in range(n):
        for j in range(n):
            A[i, j] = math.sqrt(2/(n+1)) * math.sin((i+1)*(j+1)*math.pi/(n+1))
    # xesatta = (1:n)^trasposta
    xEsatta = np.arange(1, n+1).reshape(n, 1)  #np.arange: Return evenly spaced values within a given interval, reshape: rende le liste di numeri liste di liste di 1 elem
    b = np.dot(A, xEsatta) # A x = b => x = A / b, vettore dei termini noti
    
    # LU_nopivot 
    t1 = time.perf_counter()
    P, L, U, flag = Sistemi_lineari.LU_nopivot(A)
    x, flag = Sistemi_lineari.LUsolve(L, U, P, b)
    t2 = time.perf_counter()
    tempoNoPivot.append(t1-t2)
    erroreRelativoNoPivot.append(npl.norm(x - xEsatta, 1)/npl.norm(xEsatta, 1)) 

    # LU_pivot 
    t1 = time.perf_counter()
    pivotP, pivotL, pivotU, flag = Sistemi_lineari.LU_pivot(A) 
    xpivot, flag = Sistemi_lineari.LUsolve(pivotL, pivotU, pivotP, b)
    t2 = time.perf_counter()
    tempoPivot.append(t2 - t1)
    erroreRelativoPivot.append(npl.norm(xpivot - xEsatta, 1)/npl.norm(xEsatta, 1))

    #scipy linalg method 
    t1 = time.perf_counter()
    xsc = spl.solve(A, b)
    t2 = time.perf_counter()
    tempoLib.append(t2 - t1)
    erroreRelativoLib.append(npl.norm(xsc - xEsatta, 1)/npl.norm(xEsatta, 1))

#grafici 

plt.semilogy(range(numInizio,numFine), erroreRelativoNoPivot,
             range(numInizio,numFine), erroreRelativoPivot, 
             range(numInizio,numFine), erroreRelativoLib)

plt.legend(['No pivot', 'Pivot', 'Solve'])

plt.show()

curva = np.arange(numInizio,numFine)**3 #np.arange: Return evenly spaced values within a given interval

plt.semilogy(range(numInizio,numFine), tempoNoPivot,'r-',
             range(numInizio,numFine), tempoPivot,'g-', 
             range(numInizio,numFine), tempoLib,'b-',
             range(numInizio,numFine), curva,'m-')

plt.legend(['No pivot', 'Pivot', 'Solve', 'n**3'])

plt.show()
```

## 2

```py
import numpy as np

import Sistemi_lineari as fSl

import scipy.linalg as spl

scelta = input("Scegli Matrice ")

scelta_Matrice = {
        '1': [np.array([[1,2,3],[0,0,1],[1,3,5]],dtype=float),np.array([[6],[1],[9]],dtype=float)], # dtype=float assicura che gli elementi della matrice siano float per non perdere dati con il troncamento ad interi 
        '2':[np.array([[1, 1, 0, 3], [2, 1, -1, 1],[-1, 2, 3, -1],[ 3, -1, -1 ,2]],dtype=float),np.array([[5],[3],[3],[3]],dtype=float)]
        # la seconda matrice non ha rango massimo => det = 0
}


A, b = scelta_Matrice.get(scelta) 

m, n = A.shape 

xesatta = np.ones((n, 1)) # Return a new array of given shape and type, filled with ones

P, L, U, flag,  = fSl.LU_nopivot(A)

if flag==0:
    x_nopivot, flag =fSl.LUsolve(L, U, P, b)
    print("soluzione con strategia pivotale \n", x_nopivot)
else:
    print("Sistema non risolubile senza strategia pivotale")
    
    
P_pivot, Lpivot, Upivot, flagpivot,   = fSl.LU_pivot(A)

if flagpivot==0:
    x_pivot, flag =fSl.LUsolve(Lpivot, Upivot, P_pivot, b)
    print("soluzione con strategia pivotale \n", x_pivot)
else:
    print("Sistema non risolubile con strategia pivotale")

"""
ipotesi metodo di Guass senza pivoting:
	matrici che abbiano rango massimo => determinante diverso da zero, tutte le matrici di testa con det != 0
Percio' il metodo di fattorizzazione senza pivoting fallisce, le matrici 1 e 2 non hanno rango massimo 
"""
```

## 3

```py
import numpy as np

import Sistemi_lineari as fSl

import scipy.linalg as spl

scelta = input("Scegli Matrice ")

scelta_Matrice = {
        '1':np.array([[3, 5, 7], [2, 3, 4], [5, 9, 11]],  dtype=float),  # dtype serve per obbligare a float i dati,  altrimenti in int si perderebbero i numeri 
        '2':np.array([[1,  2,  3,  4],  [2,  -4,  6,  8], [-1,  -2,  -3,  -1], [ 5,  7,  0 , 1]],  dtype=float)
}

A = scelta_Matrice.get(scelta) 

m, n = A.shape # Return the shape of an array/matrix

B = np.eye(m) # Return a 2-D array with ones on the diagonal and zeros elsewhere

#Calcolo l'inversa risolvendo n sistemi lineari ciascuno dei quali ha la stessa
#matrice e termine noto uguale all-iesima colonna della matrice identita' B 
X = fSl.solve_nsis(A, B)

"""
Sostituendo in solve_nsis la funzione con e senza pivot noto che: 
Con pivot tutto torna confrontando con scipy.linalg 
Senza pivot la seconda matrice non ha tutti i minori principali di rango massimo e l'algoritmto fallisce 
"""

print('Inversa risolvendo n sistemi lineari \n',  X)

Xpy = spl.inv(A)

print('Inversa usando scipy.linalg,  \n',  Xpy)
```

## 4

```py
import numpy as np

import Sistemi_lineari as fSl

import scipy.linalg as spl

import matplotlib.pyplot as plt

xesatta = np.array([[2],[2]])

err_rel_nopivot=[]

err_rel_pivot=[]

for k in range(2,19,2):
    A = np.array([[10.0**(-k), 1], [1, 1]])
    b = np.array([[2+10.0**(-k)], [4]])

    P,L,U,flag, = fSl.LU_nopivot(A)

    if flag==0:
        x_nopivot,flag=fSl.LUsolve(L, U, P, b)
    else:
        print("Sistema non risolubile senza strategia pivotale")
    
    err_rel_nopivot.append(np.linalg.norm(x_nopivot-xesatta, 1)/np.linalg.norm(xesatta, 1))
    
    P_pivot,Lpivot,Upivot,flagpivot, = fSl.LU_pivot(A)

    if flagpivot==0:
        x_pivot,flag=fSl.LUsolve(Lpivot,Upivot,P_pivot,b)
    else:
        print("Sistema non risolubile con strategia pivotale")
        
    err_rel_pivot.append(np.linalg.norm(x_pivot-xesatta, 1)/np.linalg.norm(xesatta, 1))
    

plt.semilogy(range(2,19,2), err_rel_nopivot, range(2,19,2), err_rel_pivot)

plt.legend(['No pivot', 'Pivot'])

plt.show()

"""
oltre il 10^8 con l'algoritmo senza pivoting aumenta notevolememnte l'errore commessso
"""
```

## 5

```py
import numpy as np

import Sistemi_lineari as fSl

import scipy.linalg as spl

scelta=input("Scegli Matrice ")

scelta_Matrice = {
        '1':[np.array([[3, 1, 1, 1],[2, 1, 0, 0],[2, 0, 1, 0],[2, 0, 0, 1]],dtype=float),
             np.array([[4],[1],[2],[4]],dtype=float),np.array([[1],[-1],[0],[2]],dtype=float)],
        '2': [np.array([[1, 0, 0, 2],[0, 1, 0, 2],[0, 0, 1, 2],[1, 1, 1, 3]],dtype=float),
              np.array([[4],[1],[2],[4]],dtype=float),np.array([[2],[-1],[0],[1]],dtype=float)]
    }


A, b, xesatta = scelta_Matrice.get(scelta) 

m, n = A.shape # Return the shape of an array/matrix

xesatta = np.ones((n,1)) # Return a new array of given shape and type, filled with ones

P, L, U, flag,  = fSl.LU_nopivot(A)

if flag==0:
    x_nopivot, flag=fSl.LUsolve(L, U, P, b)
    print("soluzione con strategia pivotale \n", x_nopivot)
else:
    print("Sistema non risolubile senza strategia pivotale")
    
max_L_nopivot = np.max(np.abs(L))
max_U_nopivot = np.max(np.abs(U))
    
P_pivot, Lpivot, Upivot, flagpivot,  = fSl.LU_pivot(A)

if flagpivot==0:
    x_pivot,flag=fSl.LUsolve(Lpivot,Upivot,P_pivot,b)
    print("soluzione con strategia pivotale \n",x_pivot)
else:
    print("Sistema non risolubile con strategia pivotale")
    
max_L_pivot=np.max(np.abs(Lpivot))

max_U_pivot=np.max(np.abs(Upivot))

print("(fatt no pivot) Massimo matrice L, ", max_L_nopivot,"Massimo matrice U (fatt no pivot) ", max_U_nopivot)

print("(fatt pivot)   Massimo matrice L,  ", max_L_pivot,"Massimo matrice U ", max_U_pivot)
```

## 6
```py
import numpy as np

import numpy.linalg as npl

import Sistemi_lineari as fSl

import matplotlib.pyplot as plt

n = 100

xe = np.ones((n, 1)) 

norm_xe = npl.norm(xe, 2)
 
v = np.random.rand(n, 1)

v = v/npl.norm(v, 2);

Q = np.eye(n)-2*np.outer(v, v.T) # eye : Return a 2-D array with ones on the diagonal and zeros elsewhere.
                                # outer: Compute the outer product of two vectors,  output: matrix 

D = np.eye(n)

xesatta = np.ones((n, 1))

erroreRelativoNoPivot = []
erroreRelativoPivot = []
indCond = []

for k in range(1, 21):
   D[n-1, n-1] = 10.0**k

   A = np.dot(Q, D)
   
   indCond.append(npl.cond(A, 2))

   b = np.dot(A, xesatta)

   P, L, U, flag,   =  fSl.LU_nopivot(A)

   if flag == 0:
        x_nopivot, flag = fSl.LUsolve(L, U, P, b)
   else:
        print("Sistema non risolubile senza strategia pivotale")
    
   erroreRelativoNoPivot.append(np.linalg.norm(x_nopivot-xesatta, 1)/np.linalg.norm(xesatta, 1))
    
   P_pivot, Lpivot, Upivot, flagpivot,   =  fSl.LU_pivot(A)

   if flagpivot==0:
        x_pivot, flag=fSl.LUsolve(Lpivot, Upivot, P_pivot, b)
   else:
        print("Sistema non risolubile con strategia pivotale")
        
   erroreRelativoPivot.append(np.linalg.norm(x_pivot-xesatta, 1)/np.linalg.norm(xesatta, 1))
   


plt.loglog(indCond, erroreRelativoNoPivot, 'ro-', indCond, erroreRelativoPivot, 'bo-')
plt.legend(['No pivot', 'Pivot'])
plt.xlabel('Indice di condizionamento')
plt.ylabel('Errore relativo sulla soluzione')
plt.show()
```

## 7

```py
import numpy as np

import numpy.linalg as npl

import Sistemi_lineari as fSl

import scipy.linalg as spl

import matplotlib.pyplot as plt

def Hankel(n):
    """
    Nell'algebra lineare, una matrice di Hankel è una matrice quadrata con diagonali costanti.
    a  b  c  d  e
    b  c  d  e  f
    c  d  e  f  g
    d  e  f  g  h
    e  f  g  h  i
    """
    A=np.zeros((n,n),dtype=float)
    for i in range(0,n):
        for k in range(i+1-n,i+1):
            if k>0:
                A[i,n-1+k-i]=2.0**(k+1)
            else:
                A[i,n-1+k-i]=2.0**(1/(2-k-1))
    return A
                               
indCond=[]

err_rel_pivot=[]

err_rel_qr=[]

for n in range(4,41,6):

   A=Hankel(n)

   indCond.append(npl.cond(A,2))

   xesatta=np.ones((n,1))

   b=np.dot(A,xesatta)
       
   P_pivot,Lpivot,Upivot,flagpivot, = fSl.LU_pivot(A)

   if flagpivot==0:
        x_pivot,flag=fSl.LUsolve(Lpivot,Upivot,P_pivot,b)
   else:
        print("Sistema non risolubile con strategia pivotale")
  
   xsolve=spl.solve(A,b)

   err_rel_pivot.append(np.linalg.norm(x_pivot-xesatta,2)/np.linalg.norm(xesatta,2))
   
   Q,R=spl.qr(A) #
   
   y=np.dot(Q.T,b) # y = Prodotto tra matrice trasposta con vettore b

   xqr,flag=fSl.Usolve(R,y)

   err_rel_qr.append(np.linalg.norm(xqr-xesatta,2)/np.linalg.norm(xesatta,2))
   
plt.plot(range(4,41,6),err_rel_pivot,'ro-',range(4,41,6),err_rel_qr,'go-')

plt.legend(['Pivot','QR'])
"""
fattorizzazione Qr e' piu stabile rispetto algoritmo con pivoting
"""
plt.ylabel('Errore relativo sulla soluzione')

plt.show()
```

