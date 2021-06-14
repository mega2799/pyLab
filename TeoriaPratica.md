# TeoriaPratica 

# Teorema Fattorizzazione LU

Una matrice A ammette fattorizzazione LU <=> le matrici minori principali hanno rango massimo, cioe' determinante != 0

`
for i in range(0,n):
	np.linalg.det(A[:i+1, :i+1])
`
