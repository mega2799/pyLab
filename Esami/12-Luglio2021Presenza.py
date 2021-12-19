import numpy as np 

import sympy as sym 

import sympy.utilities.lambdify as lambdify 

import matplotlib.pyplot as plt 

import math

import numpy.linalg as npl

from scipy.linalg import qr

# def 

x = [0 , 4, 0, 5]

y = [0, 0, 4, 6]

a0 = sym.symbols('a0') 

a1 = sym.symbols('a1')

a2 = sym.symbols('a2')

# A = np.array([[0, 0, 0 , 0], [16, 0, 4, 0], [0, 16, 0, 4], [25, 36, 5,  6]], dtype=float)

A = np.array([[0, 0, 1], [4, 0, 1], [0, 4, 1], [5, 6, 0]], dtype=float)

# A = np.array([[0, 0, 0 , 0], [16, 0, 4*a0, 0], [0, 16, 0, 4*a1], [25, 36, 5*a0,  6*a1]], dtype=float)

# b = np.array([a2, a2, a2, a2]) 

b = np.array([0, -16, -16, -61]) 

# b) 

Q, R = qr(A)

# c) 

b_tilde = np.dot(Q.T, b)

b_tilde2 = b_tilde[:3]

print(b_tilde2)

# d) 

val = npl.norm(np.dot(A, b_tilde2), 2) ** 2 

c = np.array([-b_tilde2[0]/2, -b_tilde2[1]/2])

print(c)