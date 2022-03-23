import numpy as np
import matplotlib.pyplot as plt
import time

r_blank = np.linspace(1,3,500)
theta_blank = np.linspace(0,2*np.pi,500)

r,theta = np.meshgrid(r_blank, theta_blank)

u = 1
u_temp = 0
a = 1
b = 3


def u(r, theta):
    val = 0 
    for k in range(1,50):
        n = 2*k-1
        val  += (2* np.sin(n * theta) * (r**-n - r**n))/ ( np.pi * (n**2) * (3**(-n-1) + 3**(n-1)))
    val += 3/2 * np.log(r)
    return val

solution = u(r, theta)

X, Y = r*np.cos(theta), r*np.sin(theta)

# plot the solution
fig, ax1 = plt.subplots()
cmap = plt.get_cmap('seismic')
# cmap = cmap.reversed()
cf = ax1.contourf(X,Y,solution,cmap=cmap,alpha=0.8, interpolation='gaussian')
ax1.contour(X,Y,solution,cmap="binary")
fig.colorbar(cf, ax=ax1)
ax1.set_title('Contour Plot of the analytical solution')
ax1.set_aspect("equal")
ax1.set_xlabel('Radius (m)')
ax1.set_ylabel('Radius (m)')
plt.show()

# Plot the surface (3D)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y, solution, cmap=cmap)

# Tweak the limits and add latex math labels.
ax.set_xlabel('r')
ax.set_ylabel('r')
ax.set_zlabel('u')
ax.set_title('3D plot of analytical Solution')

plt.show()