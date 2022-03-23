from grid import Grid
import numpy as np
import matplotlib.pyplot as plt
import random

ni = 5
nj = 5

K = 2

max_t = 2
delta_t = 0.5
time = np.arange(0,max_t+delta_t,delta_t)

levels = {}

def set_boundary_condition(grid, t,x,y):
    # Top
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

def initialize_b_matrix(T, inner_domain):
    
    B = np.zeros((inner_domain_size**2,inner_domain_size**2))

    for j, row in enumerate(inner_domain):
        for i, col in enumerate(row):
            k = i + (nj -2) * j

            # For checking corners
            top_left_corner = (0,0)
            top_right_corner = ((len(row) - 1),0)
            bottom_left_corner = (0,(len(row) - 1))
            bottom_right_corner = ((len(row) - 1),(len(row) - 1))

            corner = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

        pass

    return B

for t in time:

    mesh = Grid(ni,nj)
    mesh.set_origin(0,-np.pi/2)
    mesh.set_extent(np.pi/2,np.pi/2)
    mesh.generate()

    set_boundary_condition(mesh.u, t, mesh.x[0,:], mesh.y[:,0])

    if t == 0:
        set_initial_condition(mesh.u, mesh.x[1:-1,1:-1], mesh.y[1:-1,1:-1])

    levels[t] = mesh.u

# Assemble b matrix
first_layer = levels[0]
print(np.round(first_layer,3))

inner_domain = first_layer[1:-1, 1:-1]

inner_domain_size = len(first_layer) - 2
B = np.zeros((inner_domain_size**2,inner_domain_size**2))


inner_size = ni - 2
A_mat = np.zeros(((inner_size)**2, (inner_size)**2))
B_mat = np.zeros((inner_size**2,inner_size**2))

for j in range(0, inner_size):
    for i in range(0, inner_size):

        k = i + (ni - 2) * j
        A_mat[k,k]=3

        top_left_corner = (0,0)
        top_right_corner = ((inner_size - 1),0)
        bottom_left_corner = (0,(inner_size - 1))
        bottom_right_corner = ((inner_size - 1,inner_size - 1))

        corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

        # Check if i is at the middle?
        if i > 0 and i < inner_size - 1:
            A_mat[k,k+1] = 1
            A_mat[k,k-1] = 1

        # Check if it is at the leftmost?
        elif i == 0:
            # Set the coefficient of unknown east value
            A_mat[k,k+1] = 2

            if (i,j) in corners:
                if k == 0:
                    # Get the north and west BC
                    B_mat[k,k] += first_layer[0,1] + first_layer[1,0] 
                    
                else:
                    # Get the south and west BC
                    B_mat[k,k] += first_layer[-1,1] + first_layer[-2,0]
            else:
                B_mat[k,k] += first_layer[j+1,0]

        # Check if it is the rightmost?
        elif i == inner_size - 1:
            A_mat[k, k-1] = 2

            if (i,j) in corners:
                # Get the north and east boundary condition and pass it to B_mat
                # Top right corner
                if k == inner_size - 1:
                    B_mat[k,k] += first_layer[0,-2] + first_layer[1,-1]
                # Bottom right corner
                else:
                    # Get south and east boundary
                    B_mat[k,k] += first_layer[-1,-2] + first_layer[-2,-1]
            else:
                B_mat[k,k] += first_layer[j+1,-1]

         # Check if j is at the middle (in the y-direction)?
        if j > 0 and j < inner_size - 1:
            A_mat[k,k+(ni-2)] = 1
            A_mat[k,k-(ni-2)] = 1

        # Check if it is at the topmost?
        elif j == 0:
            # Set the south unknown condition
            A_mat[k,k+(ni-2)] = 1
            
            if (i,j) not in corners:
                # Get the north known boundary condition
                B_mat[k,k] += first_layer[0,i+1]

        # Check if it is at the bottom most?
        elif j == inner_size - 1:
            # Set the north unknown condition
            A_mat[k,k-(ni-2)] = 1
            
            if (i,j) not in corners:
                # Get the south known boundary condition
                B_mat[k,k] += first_layer[-1,i+1]

fig, ax = plt.subplots()
ax.spy(B_mat, precision=0)

print(np.round(B_mat,3))

plot_grid = np.arange(0,inner_size**2 + 1, 1)
offset_grid = np.arange(-0.5,inner_size**2 + 1, 1)

ax.set_xticks(plot_grid, minor=False)
ax.set_yticks(plot_grid, minor=False)
ax.set_xticks(offset_grid, minor=True)
ax.set_yticks(offset_grid, minor=True)

# ax.set_yticks(np.linspace(0,inner_size**2,1))
# ax.yaxis.grid(True, which='major')
ax.xaxis.grid(True, which='minor')
ax.yaxis.grid(True, which='minor')
plt.imshow(B_mat,interpolation='none',cmap='binary')
# plt.grid()
plt.colorbar()
plt.show()

# mat_b = initialize_b_matrix(inner_domain)


