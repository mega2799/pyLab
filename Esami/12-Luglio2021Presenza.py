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

A = np.array([[0, 0, 0 , 0], [16, 0, 4, 0], [0, 16, 0, 4]], dtype=float)

# A = np.array([[0, 0, 0 , 0], [16, 0, 4*a0, 0], [0, 16, 0, 4*a1], [25, 36, 5*a0,  6*a1]], dtype=float)

# b = np.array([a2, a2, a2, a2]) 

b = np.array([0, 0, 0, 0]) 

# b) 

Q, R = qr(A)

# c) 

b = np.array([-4, -4, 0])

