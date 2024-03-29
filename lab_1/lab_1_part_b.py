import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm


x = np.linspace(0,4)
y = np.linspace(0,1)

def f(x, y):
    return x**4 - 6*x**2*y**2 + y**4


nx, ny = (5,5)
x = np.linspace(0,1,nx)
y = np.linspace(0,np.pi,ny)
X,Y = np.meshgrid(x,y)
Z = np.zeros((nx,ny))

Z = f(X,Y)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

print(np.min(Z), np.max(Z))

plt.show()

plt.pcolormesh(X,Y,Z)
plt.show()