from sympy import *
from sympy.functions import exp
from sympy import sin
import sympy as sp

x,y = sp.symbols('x y')

u = exp(-x)*sin(y)

dudx = u.diff(x)
d2udx2 = dudx.diff(x)

dudy = u.diff(y)
d2udy2 = dudy.diff(y)

laplace = d2udx2 + d2udy2
print(laplace)

