from sympy import *
import sympy as sp

x,y = sp.symbols('x y')

u = x**4 - 6*x**2*y**2 + y**4

dudx = u.diff(x)
d2udx2 = dudx.diff(x)

dudy = u.diff(y)
d2udy2 = dudy.diff(y)

laplace = d2udx2 + d2udy2

print(dudx)
print(d2udx2)
print(dudy)
print(d2udy2)
print(laplace)