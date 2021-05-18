# Esercitazione 5

## 1

```py
import numpy as np

import matplotlib.pyplot as plt

from funzioni_Approssimazione_MQ import metodoQR

scegli_set = input('Scegli set dati\n')

scelta_dati = {
        '1': [np.array([-3.5,-3, -2, -1.5, -0.5, 0.5, 1.7, 2.5, 3]),np.array([-3.9,-4.8,-3.3,-2.5, 0.3,1.8,4,6.9,7.1])],
        '2': [np.array([-3.14,-2.4,-1.57,-0.7,-0.3,0.0,0.4,0.7,1.57]),np.array([0.02,-1,-0.9,-0.72, -0.2,-0.04,0.65, 0.67,1.1])],
        '3': [np.linspace(0,3,12),np.exp(np.linspace(0,3,12))*np.cos(np.linspace(0,3,12) * 4)+np.random.random((12,))],
        '4': [np.array([1.001,1.0012,1.0013,1.0014, 1.0015, 1.0016]),np.array([-1.2,-0.95,-0.9, -1.15,-1.1, -1])]
}

x,y = scelta_dati.get(scegli_set) 

n = int(input('Grado del polinomio di approssimazione\n'))

a = metodoQR(x,y,n) # Approssimazione ai minimi quadrati

residuo = np.linalg.norm(y-np.polyval(a,x))**2

print("Norma del residuo ",residuo)

xmin = np.min(x)

xmax = np.max(x)

xval = np.linspace(xmin,xmax,100)

p = np.polyval(a,xval)

#plt.plot(xval,p,'r-',x,y,'o')
plt.plot(xval,p,'r-',x,y,'b-o') # Congiunge i dati con linea blu

plt.legend(['Polinomio Approssimante di grado '+str(n), 'Dati'])

plt.show()
```

## 2 

```py
import numpy as np

import matplotlib.pyplot as plt

from funzioni_Approssimazione_MQ import metodoQR

x = np.array([0.0004, 0.2507, 0.5008, 2.0007, 8.0013])

y = np.array([0.0007, 0.0162,0.0288, 0.0309, 0.0310])

#Calcolo della retta di regessione
a = metodoQR(x,y,1)
residuo = np.linalg.norm(y-np.polyval(a,x))**2
print("Norma al quadrato del residuo Retta di regressione",residuo)
xmin = np.min(x)
xmax = np.max(x)
xval = np.linspace(xmin,xmax,100)
p = np.polyval(a,xval)
plt.plot(xval,p,'r-',x,y,'o')
plt.legend(['Retta di regressione', 'Dati'])
plt.show()

#Calcolo della parabola di approssimazione nel senso dei minimi quuadrati
a = metodoQR(x,y,2)
residuo = np.linalg.norm(y-np.polyval(a,x))**2
print("Norma al quadrato del residuo Polinomio di approssimazione di grado 2",residuo)
xmin = np.min(x)
xmax = np.max(x)
xval = np.linspace(xmin,xmax,100)
p = np.polyval(a,xval)
plt.plot(xval,p,'r-',x,y,'o')
plt.legend(['Polinomio di approssimazione di grado 2', 'Dati'])
plt.show()


#Calcolo della cubica di approssimazione nel senso dei minimi quuadrati
a = metodoQR(x,y,3)
residuo = np.linalg.norm(y-np.polyval(a,x))**2
print("Norma al quadraato del residuo Polinomio di approssimazione di grado 3",residuo)
xmin = np.min(x)
xmax = np.max(x)
xval = np.linspace(xmin,xmax,100)
p = np.polyval(a,xval)
plt.plot(xval,p,'r-',x,y,'o')
plt.legend(['Polinomio di approssimazione di grado 3', 'Dati'])
plt.show()
```

## 3

```py
import numpy as np

import matplotlib.pyplot as plt

from funzioni_Approssimazione_MQ import metodoQR

x = np.arange(10.0,10.6,0.1)
y = np.array([11.0320, 11.1263, 11.1339, 11.1339, 11.1993, 11.1844])

#Calcolo del polinomio di regressione di grado 4
a = metodoQR(x,y,4)
residuo = np.linalg.norm(y-np.polyval(a,x))**2
print("Norma al quadraato del residuo Polinomio di approssimazione di grado 4",residuo)
xmin = np.min(x)
xmax = np.max(x)
#grafico 
xval = np.linspace(xmin,xmax,100)
p = np.polyval(a,xval)
plt.plot(xval,p,'r-',x,y,'o')
plt.legend(['Polinomio di approssimazione di grado 4', 'Dati'])
plt.show()

xp = x.copy()
yp = y.copy()
#Perturbazione leggera
xp[1] = xp[1]+0.013
yp[1] = yp[1]-0.001

#Calcolo del polinomio di regressione di grado 4
a = metodoQR(xp,yp,4)
residuo = np.linalg.norm(y-np.polyval(a,x))**2
print("Norma al quadraato del residuo Polinomio di approssimazione di grado 4",residuo)
xmin = np.min(x)
xmax = np.max(x)
xval = np.linspace(xmin,xmax,100)
p = np.polyval(a,xval)
plt.plot(xval,p,'b-',xp,yp,'+')
plt.legend(['Polinomio di approssimazione di grado 4', 'Datipertrubati'])
plt.show()
 
"""
il metodoQR produce risultati esatti nonostante i dati sono 
perturbati perche' la matrice e'ben posta
"""
```

## Interpolazione polinomiale nella forma di Lagrange 


### 1
```py
import numpy as np

from funzioni_Interpolazione_Polinomiale import plagr

import matplotlib.pyplot as plt

scelta = input('Scegli nodi\n')

scelta_nodi={
    '1': np.arange(0,1.1,1/4), # L'intervallo da 0 a 1, di passo .25 > [0, .25, .5, .75, 1] 
    '2': np.array([-1, -0.7, 0.5, 1]) 
    } 

xnodi = scelta_nodi.get(scelta) 

n = xnodi.size

xx = np.linspace(xnodi[0],xnodi[n-1],200);

for k in range(n):
    p=plagr(xnodi,k) #Calcola il k-esimo polinomio di lagrange 
    L=np.polyval(p,xx) #valuta il polinomio in ogni punto 
    plt.plot(xnodi,np.zeros((n,)),'ro') # Nodi con pallini
    plt.plot(xnodi[k],1,'c*') 
    plt.plot(xx,L,'b-');
    plt.show()

"""
verifica che polinomio ha askjbdabjasdkjb nfaskjbafjknasdfk 
rappresentazione grafica polinomio di lagrange
                        -Lazzaro
""" 
```

### 2

```py
import numpy as np

import math 

from funzioni_Interpolazione_Polinomiale import InterpL

import matplotlib.pyplot as plt

#nodi del problema di interpolazione 
x=np.arange(0,2*math.pi+0.1,math.pi/2); # + 0.1 perche esclude l'ultimo putno altrimenti
y1=np.sin(x)
xx=np.arange(0,2*math.pi+0.1,math.pi/40);
yy1=InterpL(x,y1,xx); 


plt.plot(xx,yy1,'b--',x,y1,'*',xx,np.sin(xx),'g-');
plt.legend(['interpolante di Lagrange','punti di interpolazione','y=sin(x)']);
plt.show()

y2=np.cos(x);
yy2=InterpL(x,y2,xx);
plt.plot(xx,yy2,'r--',x,y2,'*',xx,np.cos(xx),'c-');
plt.legend(['interpolante di Lagrange','punti di interpolazione','y=cos(x)']);
plt.show()

"""
visualizzo i polinomi che interpolano le funzioni sin e cos 
di lagrange rispetto alle reali funzioni
"""
```

### 3 

```py
import numpy as np

from funzioni_Interpolazione_Polinomiale import InterpL

import matplotlib.pyplot as plt

# Nodi del problema di interpolazione 
T = np.array([-55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65])
L = np.array([3.7, 3.7,3.52,3.27, 3.2, 3.15, 3.15, 3.25, 3.47, 3.52, 3.65, 3.67, 3.52])

# Punti di valutazione per l'interpolante
xx = np.linspace(np.min(T),np.max(T),200) # Scelta intervallo per interpolazione
pol = InterpL(T,L,xx) # Calcola  il  polinomio  interpolante  in  forma  di Lagrange

# Variazione di temperatura alle latitudini L= +42, -42 
pol42 = InterpL(T,L,np.array([42]))
pol_42 = InterpL(T,L,np.array([-42]))

plt.plot(xx,pol,'b--',T,L,'r*',42,pol42,'og',-42,pol_42,'og');
plt.legend(['interpolante di Lagrange','punti di interpolazione','stima 1', 'stima2']);
plt.show()

```

### 4

```py
import numpy as np

from funzioni_Interpolazione_Polinomiale import InterpL

import matplotlib.pyplot as plt

import math

#Interpolazioni polinomi equidistanti e/o Chebishev, con relativi errori

def zeri_Cheb(a,b,n):
    t1 = (a+b)/2
    t2 = (b-a)/2
    x = np.zeros((n+1,)) # Array di zeri
    for k in range(n+1):
        x[k] = t1+t2*np.cos(((2*k+1)/(2*(n+1))*math.pi))
    return x
                          
sceltaf = input("Scegli funzione\n")
 
scelta_funzione = {
        '1': [lambda x: np.sin(x) - 2*np.sin(2*x), -math.pi, math.pi],
        '2': [lambda x: np.sinh(x), -2,2],
        '3': [lambda x: np.abs(x), -1,1],
        '4': [lambda x: 1/(1+x**2), -5,5] #Funzione di runge
}

f,a,b = scelta_funzione.get(sceltaf)

sceltap= input("Scegli tipo punti : 1(equidistanti) 2(Chebishev)\n ")

n = int(input('Grado del polinomio\n'))

scelta_punti = {
        '1': np.linspace(a,b,n+1),
        '2': zeri_Cheb(a,b,n)
        }

x=scelta_punti.get(sceltap)

# punti di valutazione per l'interpolante
xx = np.linspace(a,b,200)

y = f(x)

pol=InterpL(x,y,xx)

plt.plot(xx,pol,'b--',x,y,'r*',xx,f(xx),'m-');
plt.legend(['interpolante di Lagrange','punti di interpolazione','Funzione']);
plt.show()

r=np.abs(f(xx)-pol)
norm_inf_r=np.linalg.norm(r,np.inf)

print("Norma infinito di r ",norm_inf_r)
plt.plot(xx,r,'m-');
plt.legend(['Errore']);
plt.show()

"""
errore nullo nei punti di interpolazione
fenomeno di runge
all' aumentare del grado la funzione di interpolazione assomiglia sempre di piu
alla funzione iniziale ma non in tutte le funzioni
??????
"""
```

### 5

```py
import numpy as np

from funzioni_Interpolazione_Polinomiale import plagr

import math

# La costante di lebesgue serve a misurare il condizionamento del problema 
def zeri_Cheb(a,b,n):
    t1 = (a+b)/2
    t2 = (b-a)/2
    x = np.zeros((n+1,))
    for k in range(n+1):
        x[k] = t1+t2*np.cos(((2*k+1)/(2*(n+1))*math.pi))

    return x

# Calcolo della costante di lebesgue in [-1,1]
xx = np.linspace(-1,1,200)
LLe = np.zeros((4,1))
LLc = np.zeros((4,1))

i = 0

for n in range(5,25,5):
    # Nodi equispaziati
    xe = np.linspace(-1,1,n+1) # Nodi = grado + 1 
    # Nodi di Chebyshev 
    xc = zeri_Cheb(-1,1,n)

    Le = np.zeros((200,1))
    Lc = np.zeros((200,1))

    for l in range (n+1):        
        pe = plagr(xe,l)
        Le = Le+np.abs(np.polyval(pe,xx))
        pc = plagr(xc,l)
        Lc = Lc+np.abs(np.polyval(pc,xx))
    
    LLe[i] = np.max(Le)
    LLc[i] = np.max(Lc)
    i = i+1

print('Costante di Lebesgue con nodi equispaziati al variare di n: \n ',LLe)
 
print('Costante di Lebesgue con nodi di Chebyshev al variare di n: \n',LLc)
 
```

### 6

```py
import numpy as np

import numpy.linalg as npl

from funzioni_Interpolazione_Polinomiale import InterpL

import math

import matplotlib.pyplot as plt

f= lambda x: np.sin(2*math.pi*x)
x = np.linspace(-1,1,22) # 22 punti 
y1 = f(x) 

y2 = y1.copy()
y2 = y2 + 0.0002*np.random.random(22,) # Pertubazione

xx=np.linspace(-1,1,300)
pol1=InterpL(x,y1,xx)
pol2=InterpL(x,y2,xx)


#Funzione "normale"
plt.plot(xx,pol1,'b--',x,y1,'r*',xx,f(xx),'m-');
plt.legend(['interpolante di Lagrange','punti di interpolazione','Funzione']);
plt.show()

#Funzione perturbata
plt.plot(xx,pol2,'b--',x,y2,'r*',xx,f(xx),'m-');
plt.legend(['interpolante di Lagrange','punti di interpolazione perturbati','Funzione']);
plt.show()

#errore relativo sui dati
errrel_dati=npl.norm(y2-y1,np.inf)/npl.norm(y1,np.inf)

#errore relativo sui risultati
errrel_risultati=npl.norm(pol2-pol1,np.inf)/npl.norm(pol1,np.inf)

print("Errore relativo sui dati ",errrel_dati)
print("Errore relativo sui risultati ",errrel_risultati)

"""
l'errore sui risultati è amplificato rispetto a quella piccolissima perturbazione 
sui dati
"""
```

