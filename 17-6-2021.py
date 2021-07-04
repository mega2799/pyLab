import numpy as np

import src.Sistemi_lineari as lin

B = np.array([[.98, .02, 0, .04, 0], [.08, .93, .08, -.07, -.03], [.04, .01, .97, -.07, -.04], [.02, -.03, 0, 1.03, 0], [.07, .04, 0, -.08, 1.01]])

# a)
n = B[0].size

I = np.eye(n)

A = I - B

M = np.max(np.abs(A))

print("M < 1/5: ", M < 1/5)

# b)

res = np.zeros((n,n))

res = I 

res += A

res += np.dot(A, A) 

res += np.dot(np.dot(A,A), A)

invB = np.linalg.inv(B)

errore = np.linalg.norm(invB - res) / np.linalg.norm(invB)

print("errore nel calcolo dell inversa con potenza: ", errore) 

# c) 

#Verifico che la matrice B abbia i minori principali a rango massimo, ed in caso affermativo posso utilizzare
# il metodo di fattorizzazione di Gauss senza pivoting parziale  a perno massimo

for k in range(0, n):
    print(np.linalg.det(B[:k,:k]) != 0) 

# d) 
P, L, U, flag = lin.LU_nopivot(B)


#Calcolo l'inversa della matrice B, risolvendo n sistemi lineari aventi come matrice dei coefficienti
#la matrice B e come termine noto le n colonne della matrice identità


def Lsolve(L,b):
    m,n=L.shape

    flag=0;

    if n != m:
        print('errore: matrice non quadrata')
        flag=1
        x=[]
        return x, flag
    
     # Test singolarita'
    if np.all(np.diag(L)) != True:
         print('el. diag. nullo - matrice triangolare inferiore')
         x=[]
         flag=1
         return x, flag
    # Preallocazione vettore soluzione
    x=np.zeros((n,1))
    
    for i in range(n):
         s=np.dot(L[i,:i],x[:i]) #scalare=vettore riga * vettore colonna
         x[i]=(b[i]-s)/L[i,i]
      
     
    return x,flag


def Usolve(U,b):
    
    """
    Risoluzione con procedura backward di Rx=b con R triangolare superiore  
     Input: U matrice triangolare superiore
            b termine noto
    Output: x: soluzione del sistema lineare
            flag=  0, se sono soddisfatti i test di applicabilità
                   1, se non sono soddisfatti
    
    """ 
#test dimensione
    m,n=U.shape
    flag=0;
    if n != m:
        print('errore: matrice non quadrata')
        flag=1
        x=[]
        return x, flag
    
     # Test singolarita'
    if np.all(np.diag(U)) != True:
         print('el. diag. nullo - matrice triangolare superiore')
         x=[]
         flag=1
         return x, flag
    # Preallocazione vettore soluzione
    x=np.zeros((n,1))
    
    for i in range(n-1,-1,-1):
         s=np.dot(U[i,i+1:n],x[i+1:n]) #scalare=vettore riga * vettore colonna
         x[i]=(b[i]-s)/U[i,i]
      
     
    return x,flag

def solve_nsis_f(P,L,U,B):
  # Test dimensione  
    m,n=L.shape
    flag=0;
    if n!=m:
      print("Matrice non quadrata")
       
      return
    
    Y= np.zeros((n,n))
    X= np.zeros((n,n))
  
    
    if flag==0:
        for i in range(n):
            y,flag=Lsolve(L,np.dot(P,B[:,i]))
            Y[:,i]=y.squeeze(1)
            x,flag= Usolve(U,Y[:,i])
            X[:,i]=x.squeeze(1)
    else:
        print("Elemento diagonale nullo")
        X=[]
    return X    


if flag==0:
   X= solve_nsis_f(P, L, U, I)

errore2 = np.linalg.norm(X-invB,1)/np.linalg.norm(invB,1)

print("Errore relativo calcolo inversa metodo Soluzione n sistemi lineari ",errore2)


########################################################################################

import matplotlib.pyplot as plt

import sympy as sym 

from sympy.utilities.lambdify import lambdify 

from scipy.optimize import fsolve

import src.funzioniZeri as fzeri

x = sym.symbols('x')

f = sym.exp(x) - 4*x**2 

fx = lambdify(x, f, np)

a = -1.0 

b = 5.0

# a) 
tolx = 1e-7

xx = np.linspace(a, b, 100)

plt.plot(xx, 0*xx, xx, fx(xx))

plt.show()

# Dal grafico ci sono 3 zeri, con intervalli [-1, 0] [0, 1] [1,5] 

x0 = []

x0.append(fzeri.bisez(fx, -1,0,tolx)[0])
x0.append(fzeri.bisez(fx, 0, 1,tolx)[0])
x0.append(fzeri.bisez(fx, 1, 5,tolx)[0])

print("zeri funzione: ", x0)

# b) 

g = .5 * sym.exp(x/2) 

gx = lambdify(x, g, np)

plt.plot(xx, gx(xx), x0[0], x0[0], 'ro', x0[1], x0[1], 'ro', x0[2], x0[2], 'ro' )
plt.show()

# La funzione g puo calcolare solo la seconda e terza radice

# c,d) 

# Non si sa perche ma si aspetta la convergenza con 0.5 
x0 = .5 
x1, it, xk0 = fzeri.iterazione(gx, x0, tolx, 1000)

ordine = fzeri.stima_ordine(xk0, it)

print("ordine di convergenza: ", ordine) 

x0 = 4.5 
x1, it, xk1 = fzeri.iterazione(gx, x0, tolx, 1000)

