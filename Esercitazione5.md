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
