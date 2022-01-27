import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm


x = np.linspace(0,4)
y = np.linspace(0,1)

def f(x, y):
    return np.exp(-x)*np.sin(y)

nx, ny = (20,20)
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
X,Y = np.meshgrid(x,y)
Z = np.zeros((nx,ny))

for i in range(nx):
    for j in range(ny):
        Z[i,j] = f(X[i,j], Y[i,j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()

plt.pcolormesh(X,Y,Z)
plt.show()