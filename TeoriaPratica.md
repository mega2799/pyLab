# TeoriaPratica 

## Teorema Fattorizzazione LU

Una matrice A ammette fattorizzazione LU <=> le matrici minori principali hanno rango massimo, cioe' determinante != 0

```py
for i in range(0,n):
	np.linalg.det(A[:i+1, :i+1])
```

## Punto fisso (interpolazione)

Per trovare il punto fisso di una funzione si utilizza l'algoritmo delle iterazioni, la convergenza locale è garantita dal teorema in cui ho che | g'(x) | < 1 o in un altro modo che g'(x) sia compreso tra (-1,1)

```py
plt.plot(xx ,dg(xx))
#Disegno la retta y=1
plt.plot([-1,1],[1,1],'--')
#Disegno la retta y=-1
plt.plot([-1,1],[-1,-1],'--')
```

