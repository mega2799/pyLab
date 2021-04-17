# ES 1

## 1
Verificare numericamente l’approssimazione con rounding to even nell’intervallo [2**52, 2**53]
_____

```py
x=2**52     #  = 4503599627370496
print("x=",x)
x+1           # = 4503599627370497
print("x+1= ",x+1)
x+0.5
print("x+0.5= ",x+0.5)

          # questo numero e' 4503599627370496.5
          # ma poiche' sul segmento [2^52,2^53] ci sono solo gli interi (spacing=1)
          # x+0.5 viene approssimato con
          # 4503599627370496 
          # ossia viene arrotondato a x (per difetto)
          #  per soddisfare la regola del rounding to even

(x+1)+0.5
print("(x+1)+0.5",(x+1)+0.5)
                # questo numero e' 4503599627370497.5
                # ma viene approssimato con
                #  4503599627370498 
                # ossia viene arrotondato a (x+2) (per eccesso) 
                # per soddisfare la regola del rounding to even
```

## 2
Verificare numericamente che eps = 2**-52, spacing nell’intervallo [1, 2], e' il piu' piccolo x tale che _fl_(1+x) e' diverso da 1 
____ 

```py
p=0   
t=53

f=2**p  #numero floating point
s=2**(p+1-t) #spacing sul segmento [2^p, 2^(p+1)]=[1,2]
print("Spacing in [1,2]",s)


f1=f+s     #numero floating point successivo a f
f2=f+s/2   #risulta f
print("f+s=",f1)
print("f+s/2=",f2)

print(" risultati per differenza f+s-f=", f1-f,"f+s/2-f=",f2-f)
```

## 3
> Confrontare i risultati delle operazioni (0.3 − 0.2) − 0.1 e 0.3 − (0.2 + 0.1) e fornire una spiegazione a quanto osservato. Ripetere l’esercizio con le operazioni 0.1 ∗ (0.2 + 0.5) e 0.1 ∗ 0.2 + 0.1 ∗ 0.5 

```py

print('......esempio 1......')
x = 0.1;
y = 0.2;
z = 0.3;

ris1=(z-y)-x
ris2=z-(y+x)
print("ris1=",ris1)
print("ris2=",ris2)
#ris1= fl( fl(fl(0.3)-fl(0.2)) - fl(0.1) ) = -2^(-55) = -2.775558e-17
#ris2= fl( fl(0.3)-fl(fl(0.2) + fl(0.1)) ) = -2^(-54) = -5.551115e-17
..................................................

print('......esempio 2......')
x = 0.1;
y = 0.2;
z = 0.5;

ris1=x*(y+z) 
ris2=x*y+x*z
print("ris1=",ris1)
print("ris2=",ris2)
```
