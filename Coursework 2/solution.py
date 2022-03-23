from grid import Grid
import numpy as np
import matplotlib.pyplot as plt
import random

ni = 10
nj = 10

K = 2

max_t = 2
delta_t = 0.5
time = np.arange(0,max_t+delta_t,delta_t)

levels = {}

def set_boundary_condition(grid, t,x,y):
    grid[0,:] = random.sample(range(10, 30), ni)
    # Bottom
    grid[-1,:] = random.sample(range(10, 30), ni)
    # Left
    grid[:,0] = 0.5 * np.cos(y) * np.exp(-K*t)
    # Right 
    grid[:,-1] = t * np.cos(y)

def set_initial_condition(first_layer,x,y):
    first_layer[1:-1,1:-1] = 0.5 * np.cos(x) * np.cos(y)
    return first_layer

def plot_solution(x,y,u):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('jet')
    cf = ax.contourf(x,y, u, cmap=cmap, levels = 21)
    fig.colorbar(cf, ax=ax)
    plt.show()

for t in time:

    mesh = Grid(ni,nj)
    mesh.set_origin(0,-np.pi/2)
    mesh.set_extent(np.pi/2,np.pi/2)
    mesh.generate()

    set_boundary_condition(mesh.u, t, mesh.x[0,:], mesh.y[:,0])

    if t == 0:
        set_initial_condition(mesh.u, mesh.x[1:-1,1:-1], mesh.y[1:-1,1:-1])

    levels[t] = mesh.u

# for i, level in enumerate(levels):
#     print(np.round(levels[level],3))

# Assemble b matrix
first_layer = levels[0]
print(np.round(first_layer,3))

inner_domain = first_layer[1:-1, 1:-1]

inner_domain_size = len(first_layer) - 2
B = np.zeros((inner_domain_size**2,inner_domain_size**2))

for j, row in enumerate(inner_domain):
    for i, col in enumerate(row):
        k = i + (nj -2) * j

        top_left_corner = (0,0)
        top_right_corner = ((len(row) - 1),0)
        bottom_left_corner = (0,(len(row) - 1))
        bottom_right_corner = ((len(row) - 1),(len(row) - 1))
        corner = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

        # Top left corner
        if (i,j) == top_left_corner:
            # Get north
            north = first_layer[0,1]
            # Get west
            west = first_layer[1,0]
            B[k,k] = 100

        # Top right corner
        if (i,j) == top_right_corner:
            # Get north
            north = first_layer[0,-2]
            # Get east
            east = first_layer[1,-1]
            B[k,k] = 100

        # Bottom left corner
        if (i,j) == bottom_left_corner:
            # Get west
            west = first_layer[-2,0]
            # Get south
            south = first_layer[-1,1]
            B[k,k] = 100

        # Bottom right corner
        if (i,j) == bottom_right_corner:
            # Get east
            east = first_layer[-2,-1]
            # Get south
            south = first_layer[-1,-2]
            B[k,k] = 100
        
        elif i == 0 and (i,j) not in corner:
            # Get west 
            west = first_layer[j+1,0]

        elif i == (len(row) - 1) and (i,j) not in corner:
            # Get east
            east = first_layer[j+1,-1]
        
        elif j == 0  and (i,j) not in corner:
            # Get north
            north = first_layer[0,i+1]
            print(north)

        elif j == (len(row) - 1)  and (i,j) not in corner:
            # Get south
            south = first_layer[-1,i+1]
    



