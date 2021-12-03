# TeoriaPratica 

### norma 
```py
A = np.array([ [3, 5], [3.01, 5.01] ]) # matrice dei coefficenti

npl.norm(A, np.inf)
```

### numero di condizionamento K 
```py
numeroDiCondizionamento = npl.cond(A, np.inf)
``` 

## Teorema Fattorizzazione LU

Una matrice A ammette fattorizzazione LU <=> le matrici minori principali hanno rango massimo, cioe' determinante != 0

```py
for i in range(0,n):
	np.linalg.det(A[:i+1, :i+1])
```

## Punto fisso (interpolazione)

Per trovare il punto fisso di una funzione si utilizza l'algoritmo delle iterazioni, la convergenza locale è garantita dal teorema in cui ho che | g'(x) | < 1 o in un altro modo che g'(x) sia compreso tra (-1,1)

Avere un solo punto fisso vuol dire che g(x) intersca la bisettrice soltanto una volta

```py
plt.plot(xx ,dg(xx))
#Disegno la retta y=1
plt.plot([-1,1],[1,1],'--')
#Disegno la retta y=-1
plt.plot([-1,1],[-1,-1],'--')
```

## Errore relativo

```py
iccs = np.zeros((nval,), dtype=float) # Valore atteso

ipsilon = np.zeros((nval,), dtype=float) # Valore ottenuto

erroreRelativo = np.abs(iccs - ipsilon) / np.abs(icc) 
```

## Metodo iterativo converge qudraticamente ---> Newton

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

