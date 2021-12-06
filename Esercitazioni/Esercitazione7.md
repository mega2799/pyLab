# Esercitazione 7

## Integrazione Numerica 

### 1
```py
import sympy as sym

import Funzioni_integrazione as FI 

from sympy.utilities.lambdify import lambdify

import numpy as np

import matplotlib.pyplot as plt

scelta = input("Scegli funzione ")

x = sym.symbols('x')

scelta_funzione = {
        '1': [x**10,0.0,1.0],
        '2': [sym.asin(x),0.0,1.0],
        '3': [sym.log(1+x), 0.0,1.0]
}

fx, a, b = scelta_funzione.get(scelta)

Iesatto = float(sym.integrate(fx,(x,a,b)))

f = lambdify(x,fx,np) # Fname

N = [1, 2, 4, 8, 16, 32 ,64 ,128, 256] # Numero di sottointervalli

i = 0

InT = []

InS = []

for n in N:
    InT.append(FI.TrapComp(f,a,b,n))
    InS.append(FI.SimpComp(f,a,b,n))
  
ET = np.zeros((9,))
 
ES = np.zeros((9,))

ET = np.abs(np.array(InT)-Iesatto)/abs(Iesatto)

ES = np.abs(np.array(InS)-Iesatto)/abs(Iesatto)

plt.semilogy(N,ET,'ro-',N,ES,'b*-')

plt.legend(['Errore Trapezi Composita', 'Errore Simpson Composita'])

plt.show()
```

### 2

```py
import sympy as sym

import Funzioni_integrazione as FI 

from sympy.utilities.lambdify import lambdify

import numpy as np

import matplotlib.pyplot as plt

scelta = input("Scegli funzione ")

x = sym.symbols('x')

scelta_funzione = {
        '1': [sym.log(x),1.0,2.0],
        '2': [sym.sqrt(x),0.0,1.0],
        '3': [sym.Abs(x), -1.0,1.0]
}

fx,a,b=scelta_funzione.get(scelta)

Iesatto=float(sym.integrate(fx,(x,a,b)))

f= lambdify(x,fx,np)

tol=1e-6

IT,NT=FI.traptoll(f,a,b,tol)

print("Il valore dell'integrale esatto e' ", Iesatto)

if NT>0:
    print("Valore con Trapezi Composito Automatica ",IT," numero di suddivisoini ",NT)

IS,NS=FI.simptoll(f,a,b,tol)

if NS>0:    
    print("Valore con Simpson Composito Automatica ",IS," numero di suddivisoini ",NS)
```

### 3

```py
import sympy as sym

import Funzioni_integrazione as FI 

from sympy.utilities.lambdify import lambdify

import numpy as np

import math

import matplotlib.pyplot as plt

scelta = input("Scegli funzione ")

x = sym.symbols('x')
 
scelta_funzione = {
        '1': [sym.cos(x),0.0,2.0],
        '2': [x*sym.exp(x)*sym.cos(x**2),-2*math.pi,0],
        '3': [sym.sin(x)**(13.0/2.0)*sym.cos(x), 0.0,math.pi/2],
        '4': [sym.sin(x)**(5.0/2.0)*sym.cos(x), 0.0,math.pi/2],
        '5': [sym.sin(x)**(1.0/2.0)*sym.cos(x), 0.0,math.pi/2]
}

fx,a,b = scelta_funzione.get(scelta)

Iesatto = float(sym.integrate(fx,(x,a,b)))

print("Il valore dell'integrale esatto e' ", Iesatto)

f= lambdify(x,fx,np)

z = np.linspace(a,b,100)
plt.plot(z,f(z))
plt.show()
vett_T=[]
vett_S=[]
kvett_T=[]
kvett_S=[]
vett_NT=[]
vett_NS=[]
vett_VT=[]
vett_VS=[]
for k in range(4,11):    
    tol=10.0**(-k)
    
    IT,NT=FI.traptoll(f,a,b,tol)
   
    
    if NT>0:
        print("Valore con Trapezi Composito Automatica ",IT," numero di suddivisoini ",NT)
        vett_T.append(IT)
        kvett_T.append(k)
        vett_NT.append(NT)
        vett_VT.append(NT+1)
        
    IS,NS=FI.simptoll(f,a,b,tol)
   
    if NS>0:    
        print("Valore con Simpson Composito Automatica ",IS," numero di suddivisoini ",NS)
        vett_S.append(IS)
        kvett_S.append(k)
        vett_NS.append(NS)
        vett_VS.append(2*NS+1)
        
ET=np.abs(np.array(vett_T) - Iesatto)/np.abs(Iesatto)


ES=np.abs(np.array(vett_S)-Iesatto)/np.abs(Iesatto)

plt.semilogy(kvett_T,ET,'ro-',kvett_S,ES,'b*-')
plt.legend(['Errore Trapezi al variare di k','Errore Simpson al variare di k'])
plt.show()

plt.plot(kvett_T,vett_NT,'ro-',kvett_S,vett_NS,'b*-')
plt.legend(['Suddivsioni Trapezi al variare di k','Suddivisioni Simpson al variare di k'])
plt.show()

plt.plot(kvett_T,vett_VT,'ro-',kvett_S,vett_VS,'b*-')
plt.legend(['Valutazione funzione Trapezi al variare di k','Valutazione Funzione Simpson al variare di k'])
plt.show()
```

