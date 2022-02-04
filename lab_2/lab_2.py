import numpy as np
import matplotlib.pyplot as plt

x_max = 1
y_max = 4
x_grid_size = 100
y_grid_size = 100

def u(x,y):
    return (7/16)*np.sin(np.pi*x)*np.exp(-np.pi*y) 

x = np.linspace(0,x_max,x_grid_size)
y = np.linspace(0,y_max,y_grid_size)

# Create the mesh grid
X, Y = np.meshgrid(x,y)
Z = u(X,Y)

fig, ax = plt.subplots()
plt.contour(X, Y, Z, 20, cmap='RdGy')

plt.show()

fig2, ax2 = plt.subplots()
cmap = plt.get_cmap('PiYG')
cf = ax2.contourf(X,Y,Z,cmap=cmap)
fig2.colorbar(cf, ax=ax2)
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')


plt.show()