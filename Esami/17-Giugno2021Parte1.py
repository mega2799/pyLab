import numpy as np 

import sympy as sym 

import numpy.linalg as npl 

import matplotlib.pyplot as plt 

# def 

n = 5 

B=np.array([[0.98, 0.02, 0, 0.04, 0],
[0.08, 0.93, 0.08, -0.07, -0.03],
[0.04, 0.01, 0.97, -0.07, -0.04],
[0.02, -0.03, 0, 1.03, 0],
[0.07, 0.04, 0, -0.08, 1.01]])

# a) 

I = np.eye(n)

A = I - B 

M = np.max(np.abs(A))

print('max < 1/5 ? ' + str(M < 1/5))

# b) 

I # A^0

A # A^1 

AA = np.dot(A, A) # A^2 

AAA = np.dot(AA, A) # A^3 

somma = I + A + AA + AAA

# print(somma)

# print(npl.inv(B))

# sono coincidenti, uguali

# c) 

# ammette fattorizzazione LU senza pivoting se i determinanti dei suoi minori sono tutti positivi

for i in range(n):
    print(npl.det(B[:i+1, :i+1]) > 0)

# la matrice la ammette 

# d) 
 
def LU_nopivot(A):
    """
    % Fattorizzazione PA=LU senza pivot   versione vettorizzata
    In output:
    L matrice triangolare inferiore
    U matrice triangolare superiore
    P matrice identità
    tali che  LU=PA=A
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
       #Test pivot 
          if U[k,k]==0:
            print('elemento diagonale nullo')
            L,U,P,flag=[],[],[],1 
            return P,L,U,flag

  #     Eliminazione gaussiana
          U[k+1:n,k]=U[k+1:n,k]/U[k,k]                                   # Memorizza i moltiplicatori	  
          U[k+1:n,k+1:n]=U[k+1:n,k+1:n]-np.outer(U[k+1:n,k],U[k,k+1:n])  # Eliminazione gaussiana sulla matrice
     
  
    L=np.tril(U,-1)+np.eye(n)  # Estrae i moltiplicatori 
    U=np.triu(U)           # Estrae la parte triangolare superiore + diagonale
    return P,L,U,flag

P, L, U, flag = LU_nopivot(B)

# e) 

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


#Calcolo l'inversa della matrice B, risolvendo n sistemi lineari aventi come matrice dei coefficienti
#la matrice B e come termine noto le n colonne della matrice identità
if flag==0:
   X= solve_nsis_f(P,L,U,I)

erroreLU = npl.norm(X - npl.inv(B), 1) / npl.norm(npl.inv(B), 1)

erroreAppr = npl.norm(AAA - npl.inv(B), 1) / npl.norm(npl.inv(B), 1)

print(erroreLU, erroreAppr)