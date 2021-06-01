# Esercitazione 6

## 1
```py
from scipy.fft  import fft

import math

import numpy as np

import matplotlib.pyplot as plt

sceltaf = input("Scegli funzione ")
 
scelta_funzione = {
        '1': [lambda x: np.sin(x)-2*np.sin(2*x),-math.pi, math.pi],
        '2': [lambda x: np.sinh(x),-2,2],
        '3': [lambda x: np.abs(x), -1,1],
        '4': [lambda x: 1/(1+x**2), -5,5] # Funzione di Runge
}

f,A,B=scelta_funzione.get(sceltaf)

"""
La trasfomata di fourier si puo utilizzare se le funzioni sono definite nell intervallo 
0,2pigreco con punti equidistanti zero escluso [0, 2pigreco) 
Quindi gli intervalli di scelta_funzione vanno sistemati
""" 

n = int(input("Introduci il valore di n "))

step = (B-A)/(n+1)

# Costruisco n+1 punti equidistanti in [A,B), (l'estremo B viene quindi escluso)
x = np.arange(A,B,step)
# Mappo i punti dell'intervallo [A,B) in [l,r)  = [0,2pigreco)
l = 0
r = 2*math.pi

xm = (x-A)*(r-l)/(B-A)+l #Formula per mappare in un intervallo in cui la trasfomata è definita
# xm e' un vettore che non utilizziamo xD 
 
if n%2==0:
    m=n//2
else:
    m=(n-1)//2
# m e' un indice

y = f(x)

c = fft(y) # Coefficenti di Fourier

a = np.zeros((m+2,))

b = np.zeros((m+2,))

a0 = c[0]/(n+1)

a[1:m+1] = 2*c[1:m+1].real/(n+1)    

b[1:m+1] = -2*c[1:m+1].imag/(n+1)
 
if n%2==0:
    a[m+1]=0
    b[m+1]=0
    
else:
    a[m+1]=c[m+1]/(n+1) 
    b[m+1]=0
    
pol = a0*np.ones((100,))

z = np.linspace(A,B,100 )

zm = (z-A)*(r-l)/(B-A)+l

for k in range(1,m+2):
   pol = pol+a[k]*np.cos(k*zm)+b[k]*np.sin(k*zm) # Formula sommatoria per interpolare il polinomio 

plt.plot(z,pol,'r',x ,y ,'o',z ,f(z),'b')

plt.show()
```

## 2 
```py
from scipy.fft  import fft

import math

import numpy as np

import matplotlib.pyplot as plt

def gradino(x):
    # Definire dati su cui lavorare (interpolare)
    if x<-1 or x>1:
        f=1
    else:
        f=0
    return f

A=-3

B=3

l=0

r=2*math.pi

n=int(input("Introduci il valore di n "))

step=(B-A)/(n+1)

#Costruisco n+1 punti equidistanti in [A,B), (l'estremo B viene quindi escluso)
x=np.arange(A,B,step)
#Mappo i punti dell'intervallo [A,B) in [0,2pigreco)
xm=(x-A)*(r-l)/(B-A)+l
 
if n%2==0:
    m=n//2
else:
    m=(n-1)//2

y=np.zeros((n+1,))

for i in range(0,n+1):
    y[i]=gradino(x[i])

c=fft(y)

a=np.zeros((m+2,))

b=np.zeros((m+2,))

a0= c[0]/(n+1)

a[1:m+1]=2*c[1:m+1].real/(n+1)    

b[1:m+1]=-2*c[1:m+1].imag/(n+1) 

if n%2==0:
    a[m+1]=0
    b[m+1]=0
    
else:
    a[m+1]=c[m+1]/(n+1) 
    b[m+1]=0
    
pol=a0*np.ones((100,))

z=np.linspace(A,B,100 )

zm=(z-A)*(r-l)/(B-A)+l

for i in range(1,m+2):
   pol= pol+a[i]*np.cos(i*zm)+b[i]*np.sin(i*zm)
   plt.plot(z,pol,'r',x  ,y ,'o')
   plt.show()

"""
uguale all es1 ma qui visualizzo i grafici mano a mano che viene calcolato il 
polinomio di interpolazione
"""
```
## 3

```py
from scipy.fft  import fft

import math

import numpy as np
 
import matplotlib.pyplot as plt

'''
Il problema fornisce n+1 (xi,yi), i=0,.n n=9 misurazioni del flusso sanguigno
attraverso una sezione dell’arteria carotide durante un battito cardiaco.---
ad istanti di tempo equistanti con step 1/10
Gli istanti appartengono all'intervallo [0,1)
'''

n=9
A=0
B=1
l=0
r=2*math.pi

step=(B-A)/(n+1)

x=np.arange(A,B,step)
 
y =np.array([3.7, 13.5, 5, 4.6, 4.1, 4.5, 4, 3.8, 3.7, 3.7])

#Mappatura dell'intervallo [A,B) in [0,2*pigreco)

xm=(x-A)*(r-l)/(B-A)+l
 
if n%2==0:
    m=n//2
else:
    m=(n-1)//2


    
#Calcolo i coefficienti di Fourier della sequenza di dati y
c = fft(y)

#Da essi ricavo i coefficienti ak, k=0,..m+1 e bk, k=1,..m+1 della funzione trigonometrica interpolabnte
c=fft(y)
a=np.zeros((m+2,))
b=np.zeros((m+2,))
a0= c[0]/(n+1)
a[1:m+1]=2*c[1:m+1].real/(n+1)    
b[1:m+1]=-2*c[1:m+1].imag/(n+1) 

 
if n%2==0:
    a[m+1]=0
    b[m+1]=0
    
else:
    a[m+1]=c[m+1]/(n+1) 
    b[m+1]=0
    
    
pol = a0*np.ones((100,))
z = np.linspace(A,B,100 )
zm = (z-A)*(r-l)/(B-A)+l


for i in range(1,m+2):
   pol = pol+a[i]*np.cos(i*zm)+b[i]*np.sin(i*zm)
  

plt.plot(z,pol,'r',x  ,y ,'o')
```


## Filtraggio di un segnale con Fourier

```py
from scipy.fft  import fft, ifft

from scipy.fftpack import fftshift, ifftshift

import math

import numpy as np

import matplotlib.pyplot as plt


f = lambda x: np.sin(2*math.pi*5*x)+np.sin(2*math.pi*10*x)
noise = lambda x: 2*np.sin(2*math.pi*30*x)

T = 2     #Durata del segnale
Fs = 100  # Frequenza di campionamento nel dominio del tempo: Numero di campioni al secondo (maggiore uguale del doppio della freqeunza massima nel dominio delle frequenze
        #(wmax) presente nel segnale)
dt = 1/Fs # Passo di campionamento nel dominio del tempo
N = T*Fs  #Numero di campioni: durata in secondi per numero di campioni al secondo

#Campionamento del dominio temporale
t = np.linspace(0,T,N)

#Campionamento del segnale rumoroso
y = f(t)+noise(t)
plt.plot(t,y,'r-')
plt.title('Segnale rumoroso')
plt.show()
plt.plot(t,f(t),'b-')
plt.title('Segnale esatto')
plt.show()

#Passo di campionamento nel dominio di Fourier (si ottiene dividendo per N l'ampiezza del range che contiene le frequenze)
delta_u = Fs/N
freq = np.arange(-Fs/2,Fs/2,delta_u)  #Il range delle frequenza varia tra -fs/2 ed fs/2
c = fftshift(fft(y))


plt.plot(freq,np.abs(c))
plt.title('Spettro Fourier segnale rumoroso')
plt.show()
ind = np.abs(freq)> 10.0

#Annulliamo i coefficienti di Fourier esterni all'intervallo di frequenze [-10,10]
c[ind] = 0
plt.plot(freq,np.abs(c))
plt.title('Spettro Fourier segnale Filtrato')
plt.show()
#Ricostruiamo il segnale a partire dai coefficienti du Fourier filtrati
rec=ifft(ifftshift(c))
plt.plot(t,rec,t,f(t))
plt.legend(['Segnale filtrato', 'Segnale originale'])
```
