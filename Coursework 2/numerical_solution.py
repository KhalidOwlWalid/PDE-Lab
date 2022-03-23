import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

alpha = 2.0
K = 2

steps = 5

x_min = 0
x_max = np.pi/2

y_min = -np.pi/2
y_max = np.pi/2

delta_x = (x_max - x_min)/steps
delta_y = (y_max - y_min)/steps

x = np.linspace(x_min,x_max,steps)
y = np.linspace(y_min,y_max,steps)

delta_x = x[1] - x[0]
delta_y = x[1] - x[0]

print(delta_x)

# Calculated params
max_t = 2
delta_t = 1
t = np.arange(0,max_t+delta_t,delta_t)

print(t)

def function(x,y):
    return 0.5 * np.cos(x) * np.cos(y)

def set_boundary_conditions(levels):
   
   for i, level in enumerate(levels):
        # Top
        level[0,:] = 0
        # Bottom
        level[-1,:] = 0
        # Left
        level[:,0] = 0.5 * np.cos(y) * np.exp(-K*t[i])
        # Right 
        level[:,-1] = t[i] * np.cos(y)

def set_initial_conditions(first_layer, function):
    
    X, Y = np.meshgrid(x,y)
    first_layer = function(X,Y)
    
    return first_layer

def assemble_b_matrix(level):
    pass


# Initialize the grid with layers of time
u = np.zeros((len(t), len(x), len(y)))

# Set intial conditions
u[0] = set_initial_conditions(u[0], function)

# Set the boundaries of the first layer
set_boundary_conditions(u)


# Plot
print(np.round(u,3))
fig, ax = plt.subplots()
cmap = plt.get_cmap('jet')
ax.axis('equal')
ax.set_aspect('equal', 'box')
cf = ax.contourf(x,y, u[0], cmap=cmap, levels = 21)
fig.colorbar(cf, ax=ax)
ax.set_aspect('equal', adjustable='box')
plt.show()






