import numpy as np
import sympy as sym
import src.funzioniZeri 
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from scipy.optimize import fsolve

def iterazione(gname, x0, tolx, nmax):
    xk = []
    xk.append(x0)
    x1 = gname(x0)
    d = x1 - x0
    it = 1
    xk.append(x1)
    xi = 0
    while it < nmax and abs(d) >= tolx * abs(xi):
        xi = gname(x1)
        d = xi - x1
        xk.append(xi)
        x1 = xi
        it += 1
        if it == nmax:
            print('nmax raggiunto')
            return xi, it, xk
    return xi, it, xk

def stima_ordine(xk, it):
    p = []
    for k in range(it - 3):
        num = np.log(abs(xk[k+2]) - abs(xk[k+3]) / abs(xk[k+1]) - abs(xk[k+2])) 
        den = np.log(abs(xk[k+1]) - abs(xk[k+2]) / abs(xk[k]) - abs(xk[k+1]))
        p.append(num/den)

    return p[-1]


tolx=1e-7

nmax=1000

f= lambda x: np.tan(3/2*x)-2*np.cos(x)-x*(7-x)

#Utilizzo il metodo fsolve di scipy.optimize per calcolare lo zero alfa della funzione f,
#prende in input l'iterato iniziale x0

x0=0.0
alfa=fsolve(f, x0)
print("Lo zero della funzione e' ",alfa)
#Disegno: l'asse x e la funzione f valutata in un intervallo opportuno
xx=np.linspace(-1.0,1.0,100)
plt.plot(xx,0*xx,xx,f(xx),alfa,0,'ro')
plt.legend(['Funzione f', 'zero'])
plt.show()

'''
Definisco la funzione g in formato simbolico perchè poi utilizzo la funzione diff
di sympy per calcolare l'espressione analitica della derivata prima'
'''
x=sym.symbols('x')

#Considero la funzione g indicata dalla traccia del compito
gx=sym.tan(3/2*x)-2*sym.cos(x)-x*(6-x)

#Disegno la funzione g(x) e la bisettrice y=x
g=lambdify(x,gx,np)
plt.plot(xx,xx,xx,g(xx))
plt.title('funzione g(x) e y=x')
plt.show()

#Calcolo la derivata prima di gx espressione simbolica tramite la funzione diff del modulo sym 
dgx=sym.diff(gx,x,1)


dg=lambdify(x,dgx,np)

#Disegno la funzione dg(x) 

#Posso giustifcare la convergenza del procedimento iterativo guardando la derivata prima di g(x)
#in un intorno della soluzione: il metodo genera una successione di iterati convergenti alla radice alfa
# ed appartenenti a questo intorno se |g'(x)|< 1 in un intorno della soluzione

 
plt.plot(xx ,dg(xx ))
plt.plot(alfa,0,'ro')
#Disegno la retta y=1
plt.plot([-1,1],[1,1],'--')
#Disegno la retta y=-1
plt.plot([-1,1],[-1,-1],'--')
plt.title('funzione dg(x) proposta dalla traccia - Ipotesi per la convergenza non soddisfatte')
 
plt.legend(['Grafico derivata prima di g1 (x)', 'Zero', 'Retta y=1', 'Retta y=-1'])
plt.show()

#Dal grafico vedo che per la funzione g proposta dalla traccia
#non sono soddisfatte le ipotesi del teorema di convergenza locale


#Ricavo la funzione gx per la quale ci sia convergenza
gx1= (sym.tan(3.0/2.0*x)-2*sym.cos(x)+x**2)/7
#Disegno la funzione g(x) e la bisettrice y=x
g1=lambdify(x,gx1,np)
plt.plot(xx,xx,xx,g1(xx))
plt.title('funzione g1(x) ricavata e y=x')
plt.show()

#Calcolo la derivata prima di gx espressione simbolica tramite la funzione diff del modulo sym 
dgx1=sym.diff(gx1,x,1)


dg1=lambdify(x,dgx1,np)

#Disegno la funzione dg1(x) 

#Posso giustifcare la convergenza del procedimento iterativo guardando la derivata prima di g(x)
#in un intorno della soluzione: il metodo genera una successione di iterati convergenti alla radice alfa
# ed appartenenti a questo intorno se |g'(x)|< 1 in un intorno della soluzione

 
plt.plot(xx ,dg1(xx ))
plt.plot(alfa,0,'ro')
#Disegno la retta y=1
plt.plot([-1,1],[1,1],'--')
#Disegno la retta y=-1
plt.plot([-1,1],[-1,-1],'--')
plt.title('funzione dg1(x) Ricavata - Ipotesi di convergenza soddisfatte')
plt.legend(['Grafico derivata prima di g1 (x)', 'Zero', 'Retta y=1', 'Retta y=-1'])
plt.show()

'''
Dal grafico vedo che per la funzione g  ceh ho ricavato
soddisfatte le ipotesi del teorema di convergenza locale: esiste un intorno della soluzione per cui
|g'(x)|<1 

'''

#Utilizzo il metodo di iterazione funzionale per calcolare il punto fisso di g1
x1,it,xk=iterazione(g1,x0,tolx,nmax)
print('iterazioni= {:d}, soluzione={:e} \n\n'.format(it,x1))

#Calcolo l'ordine del metodo
ordine_iter= stima_ordine(xk,it)
#Essendo il metodo con ordine di convergenza lineare, la costante asintotica di convergenza è data
#da |g'(alfa)| dove alfa è la radice.

 
print("Iterazione it={:d}, ordine di convergenza {:e}".format(it,ordine_iter ))

plt.plot(range(it+1),xk)
