# ES 1

 1 

`
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
`
