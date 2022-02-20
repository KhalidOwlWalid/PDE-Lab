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
        val  += ((2/(np.pi*n**2*(3**(-n-1) + 3**(n-1))))*(r**(-n) - r**(n) * np.sin(n * theta)))
    val += 3/2 * np.log(r)
    return val

solution = u(r, theta)

X, Y = r*np.cos(theta), r*np.sin(theta)

# plot the solution
fig, ax1 = plt.subplots()
cmap = plt.get_cmap('jet')
cf = ax1.contourf(X,Y,solution,cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('Graph')
ax1.set_aspect("equal")
plt.show()

# Plot the surface (3D)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y, solution, cmap=plt.cm.jet)

# Tweak the limits and add latex math labels.
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

plt.show()