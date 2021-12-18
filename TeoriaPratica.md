# TeoriaPratica 

### numero di condizionamento K 
```py
numeroDiCondizionamento = npl.cond(A, np.inf)
``` 
### array di numpy 
```py
arr = np.zeros((n,), dtype=float) 
``` 
### disegnare assi grafico 
```py
plt.plot(xx, f(xx))
plt.axhline(y=0, color='r', ls='--')
plt.axvline(x=0,color='r', ls='--' )
plt.show()
```
## Teorema Fattorizzazione LU

Una matrice A ammette fattorizzazione LU <=> le matrici minori principali hanno rango massimo, cioe' determinante != 0

```py
for i in range(0,n):
	np.linalg.det(A[:i+1, :i+1])
```

## Punto fisso (interpolazione)

Per trovare il punto fisso di una funzione si utilizza l'algoritmo delle iterazioni, la convergenza locale è garantita dal teorema in cui ho che | g'(x) | < 1 o in un altro modo che g'(x) sia compreso tra (-1,1)

Avere un solo punto fisso vuol dire che g(x) intersca la bisettrice soltanto una volta, vedere [esame](Esami/15-Gennaio2021.py) per chiarimento.

## Errore relativo

```py
iccs = np.zeros((nval,), dtype=float) # Valore atteso

ipsilon = np.zeros((nval,), dtype=float) # Valore ottenuto

erroreRelativo = np.abs(iccs - ipsilon) / np.abs(icc) 
```

## Metodo iterativo converge qudraticamente ---> Newton modificato con m = molteplicita soluzione

## Risolvere equazione

```py
from sympy.utilities.lambdify import lambdify
from scipy.optimize import fsolve

x=sym.symbols('x')
fx= x-1/3*sym.sqrt(30*x-25)
x0=4
f=lambdify(x,fx,np)
alfa=fsolve(f,x0)
```

## Calcolo dell'inversa matrice B

risolvendo n sistemi lineari aventi come matrice dei coefficenti la matrice B e termine noto le n colonne della matrice identità

```py
P,L,U,flag=LU_nopivot(B)

if flag==0:
   X= solve_nsis_f(P,L,U,I)
```

## Fattorizzazione di Cholesky 

Una matrice ammette fattorizzazione di Cholesky se simmetrica e definita positiva.

Verifico sia positiva trovando gli autovalori e verificando che siano tutti positivi

```py 
import numpy.linalg as npl 

npl.egivals(A) # vettori autovalori
``` 

### Norma 2, infinito, 1 

```py 
import numpy.linalg as npl 

npl.norm(A) 
npl.norm(A, np.inf)
npl.norm(A, 1)
``` 

### inversa matrice 
```py 
import numpy.linalg as npl 

npl.inv(B)
```

### matrice trasposta

```
A = np.array([ [10, -4, 4, 0], [-4, 10, 0, 2], [4, 0, 10, 2], [0, 2, 2, 0]],dtype=float)

A.T
``` 

### norme

norma infinito: il max di ogni elemento della matrice.  
norma 2 = math.sqrt(A * At) oppure sommatoria di ogni elemento al quadrato.  
norma 1: sommatoria di valore assoluto di ogni elemento della matrice.

```
npl.norm(B, 1)
npl.norm(B, 2)
npl.norm(B, np.inf)
```