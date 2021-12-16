import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

import numpy.linalg as npl

from scipy.optimize import fsolve
# def 

Z = lambda a: np.array([[11+a, 10 + a, 14 +a], [12 +a , 11 +a, a -13], [14 + a, 13 + a, a -66]])

Zmeno1 = lambda a: np.array([[-55* a -557, 83 *a + 842, -28*a - 284] , [55*a +610 , -83*a -922, 28*a + 311], [2, -3, 1]]) 

# a) norma infinito e il max di ogni elemento della matrice, norma 2 = math.sqrt(A * At) oppure sommatoria di ogni elemento al quadrato 
        # norma 1: sommatoria di valore assoluto di ogni elemento della matrice
# TODO aggiungere in Teoria pratica le norme 

def normaInf(v):
    n, m = v.shape
    el = v[0,0]
    for i in range(n):
        for j in range(m):
            if v[i, j] > el:
                el = v[i,j]
    return el


print(normaInf(Z(30)))

print(normaInf(Zmeno1(30)))

# b) numero di condizionamento infinito, sembrerebbe essere il prodotto tra normaInf(A) e normaInf(A^-1)

KinfZ = lambda x : normaInf(Z(x)) * normaInf(Zmeno1(x))

vk = []

for i in range(30, 10000):
    vk.append(KinfZ(i))

plt.plot(range(30, 10000), vk)
#plt.plot(xx, KinfZ(xx))

plt.show()

# c) 

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


def Lsolve(L,b):
    """  
    Risoluzione con procedura forward di Lx=b con L triangolare inferiore  
     Input: L matrice triangolare inferiore
            b termine noto
    Output: x: soluzione del sistema lineare
            flag=  0, se sono soddisfatti i test di applicabilità
                   1, se non sono soddisfatti
    """
#test dimensione
    m,n=L.shape
    flag=0;
    if n != m:
        print('errore: matrice non quadrata')
        flag=1
        x=[]
        return x, flag
    
     # Test singolarita'
    if np.all(np.diag(L)) != True:  #all confronta tutti i numeri con zero
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


def swapRows(A,k,p):
    A[[k,p],:] = A[[p,k],:]
    
    
def LU_pivot(A):
    """
    % Fattorizzazione PA=LU con pivot 
    In output:
    L matrice triangolare inferiore
    U matrice triangolare superiore
    P matrice di permutazione
    tali che  PA=LU
    """
    # Test dimensione
    m,n=A.shape
    flag=0;
    if n!=m:
      print("Matrice non quadrata")
      L,U,P,flag=[],[],[],1 
      return P,L,U,flag
  
    P=np.eye(n);
    U=A.copy();
 # Fattorizzazione
    for k in range(n-1):
       #Scambio di righe nella matrice U e corrispondente scambio nella matrice di permutazione per
       # tenere traccia degli scambi avvenuti
       
       #Fissata la colonna k-esima calcolo l'indice di riga p a cui appartiene l'elemento di modulo massimo a partire dalla riga k-esima
          p = np.argmax(abs(U[k:n,k])) + k
          if p != k:
              swapRows(P,k,p)
              swapRows(U,k,p)

  #     Eliminazione gaussiana
          U[k+1:n,k]=U[k+1:n,k]/U[k,k]                                   # Memorizza i moltiplicatori	  
          U[k+1:n,k+1:n]=U[k+1:n,k+1:n]-np.outer(U[k+1:n,k],U[k,k+1:n])  # Eliminazione gaussiana sulla matrice
     
  
    L=np.tril(U,-1)+np.eye(n)  # Estrae i moltiplicatori 
    U=np.triu(U)           # Estrae la parte triangolare superiore + diagonale
    return P,L,U,flag



def solve_nsis(A,B):
  # Test dimensione  
    m,n=A.shape
    flag=0;
    if n!=m:
      print("Matrice non quadrata")
       
      return
    
    Y= np.zeros((n,n))
    X= np.zeros((n,n))
    P,L,U,flag= LU_pivot(A)
    
    if flag==0:
        for i in range(n):
            y,flag=Lsolve(L,np.dot(P,B))            # QUI HO CAMBIATO B PER POTER FAR FUNZIONARE, MA CHE ESERCIZIO DI CACCAAAAAA !!!!!
            Y[:,i]=y.squeeze(1) # Elimina una dimensione, rendendolo un array 
            x,flag= Usolve(U,Y[:,i])
            X[:,i]=x.squeeze(1)
    else:
        print("Elemento diagonale nullo")
        X=[]
    return X    

# d) 

B = lambda a: np.array([3*a + 35,  3*a +10, 3*a -39])

print(solve_nsis(Z(1e7), B(1e7))) # ES DI CACCA, DOVEVAMO CAMBIARE PURE GLI ALGORITMI.........

# e) 

print(fsolve(Z(1e7), B(1e7)))