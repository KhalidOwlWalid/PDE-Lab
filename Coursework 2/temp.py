import numpy as np
from sympy import N
from grid import Grid
import matplotlib.pyplot as plt
from numpy.linalg import solve
import scipy.sparse.linalg as LA

N = 5

ni = N
nj = N

t_min = 0
t_max = 2
dt = 0.0003
time = np.arange(t_min, t_max + dt, dt)

quiet = False

def set_boundary_condition(u_grid, t,x,y, K=2):
    # Top
    u_grid[0,:] = 0
    # Bottom
    u_grid[-1,:] = 0
    # Left
    u_grid[:,0] = 0.5 * np.cos(y) * np.exp(-K*t)
    # Right 
    u_grid[:,-1] = t * np.cos(y)

def set_initial_condition(first_layer,x,y):
    first_layer[1:-1,1:-1] = 0.5 * np.cos(x) * np.cos(y)
    return first_layer

def plot_solution(x,y,u,t_cur):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('jet')
    cf = ax.contourf(x,y, u, cmap=cmap, levels = 21)
    ax.set_title(f"Solution for {t_cur}")
    fig.colorbar(cf, ax=ax)
    plt.show()

def spy_matrix(matrix, inner_grid_size):

    if inner_grid_size > 10:
        print("Matrix too large to view")
        return 

    fig, ax = plt.subplots()
    ax.spy(matrix, precision=0)

    plot_grid = np.arange(0,inner_grid_size**2 + 1, 1)
    offset_grid = np.arange(-0.5,inner_grid_size**2 + 1, 1)

    ax.set_xticks(plot_grid, minor=False)
    ax.set_yticks(plot_grid, minor=False)
    ax.set_xticks(offset_grid, minor=True)
    ax.set_yticks(offset_grid, minor=True)

    # ax.set_yticks(np.linspace(0,inner_size**2,1))
    # ax.yaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid(True, which='minor')
    plt.imshow(matrix,interpolation='none',cmap='binary')
    plt.colorbar()

    if not quiet:
        print(np.round(matrix, 3))

    plt.show()

def initialize_matrix(cur_layer,inner_grid_size,a,b,c):

    A_mat = np.zeros(((inner_grid_size)**2, (inner_grid_size)**2))

    # Remember that B_mat should be converted to a 1 x inner_grid_size matrix 
    # TODO(Khalid): Use B_mat.diagonal() to extract the B_matrix
    B_mat = np.zeros((inner_grid_size**2,inner_grid_size**2))

    for j in range(0, inner_grid_size):
        for i in range(0, inner_grid_size):

            k = i + (ni - 2) * j
            A_mat[k,k] = c

            top_left_corner = (0,0)
            top_right_corner = ((inner_grid_size - 1),0)
            bottom_left_corner = (0,(inner_grid_size - 1))
            bottom_right_corner = ((inner_grid_size - 1,inner_grid_size - 1))

            corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

            # Check if i is at the middle?
            if i > 0 and i < inner_grid_size - 1:
                A_mat[k,k+1] = b
                A_mat[k,k-1] = b

            # Check if it is at the leftmost?
            elif i == 0:
                # Set the coefficient of unknown east value
                A_mat[k,k+1] = b

                if (i,j) in corners:
                    if k == 0:
                        # Get the north and west BC
                        B_mat[k,k] += cur_layer[0,1] + cur_layer[1,0] 
                        
                    else:
                        # Get the south and west BC
                        B_mat[k,k] += cur_layer[-1,1] + cur_layer[-2,0]
                else:
                    B_mat[k,k] += cur_layer[j+1,0]

            # Check if it is the rightmost?
            elif i == inner_grid_size - 1:
                A_mat[k, k-1] = b

                if (i,j) in corners:
                    # Get the north and east boundary condition and pass it to B_mat
                    # Top right corner
                    if k == inner_grid_size - 1:
                        B_mat[k,k] += cur_layer[0,-2] + cur_layer[1,-1]
                    # Bottom right corner
                    else:
                        # Get south and east boundary
                        B_mat[k,k] += cur_layer[-1,-2] + cur_layer[-2,-1]
                else:
                    B_mat[k,k] += cur_layer[j+1,-1]

            # Check if j is at the middle (in the y-direction)?
            if j > 0 and j < inner_grid_size - 1:
                A_mat[k,k+(ni-2)] = b
                A_mat[k,k-(ni-2)] = b

            # Check if it is at the topmost?
            elif j == 0:
                # Set the south unknown condition
                A_mat[k,k+(ni-2)] = a
                
                if (i,j) not in corners:
                    # Get the north known boundary condition
                    B_mat[k,k] += cur_layer[0,i+1]

            # Check if it is at the bottom most?
            elif j == inner_grid_size - 1:
                # Set the north unknown condition
                A_mat[k,k-(ni-2)] = a
                
                if (i,j) not in corners:
                    # Get the south known boundary condition
                    B_mat[k,k] += cur_layer[-1,i+1]

    # In later stage, please return B_mat.diagonal()
    return A_mat, B_mat

def get_b_mat(prev_layer,inner_grid_size):
    
    b_mat = np.zeros((inner_grid_size**2, inner_grid_size**2))

    for j in range(1, inner_grid_size+1):
        # Offset it by 1 to point to inner grid row

        # the i here will overlap the i value so please be careful
        for i in range(1, inner_grid_size+1):
            # Offset it by 1 to point to inner grid column
            # Why j = 1? Because we already offset it on the for loop above so it cannot be 0 anymore

            # Get all values
            # Format : b_mat[k,k] += origin + north + south + east + west
            sum = prev_layer[j,i] + prev_layer[j-1,i] + prev_layer[j+1,i] + prev_layer[j,i+1]  + prev_layer[j,i-1] 

            m = i - 1
            n = j - 1

            k = m + inner_grid_size * n

            b_mat[k,k] += sum

    return b_mat

def reset_layer(new_cur_layer,t):
    new_cur_layer = Grid(ni,nj)
    new_cur_layer.set_origin(0,-np.pi/2)
    new_cur_layer.set_extent(np.pi/2,np.pi/2)
    new_cur_layer.generate()

    # Set the boundary condition
    set_boundary_condition(new_cur_layer.u, t, new_cur_layer.x[0,:], new_cur_layer.y[:,0])

    return new_cur_layer

# Initiate the grid for the first layer
first_layer = Grid(ni,nj)
first_layer.set_origin(0,-np.pi/2)
first_layer.set_extent(np.pi/2,np.pi/2)
first_layer.generate()

# Set the initial condition of the first layer
set_boundary_condition(first_layer.u, time[0], first_layer.x[0,:], first_layer.y[:,0])
set_initial_condition(first_layer.u, first_layer.x[1:-1,1:-1], first_layer.y[1:-1,1:-1])

# Initiate the grid for second layer
second_layer = Grid(ni,nj)
second_layer.set_origin(0,-np.pi/2)
second_layer.set_extent(np.pi/2,np.pi/2)
second_layer.generate()

# Set the boundary conditions for the second layer
set_boundary_condition(second_layer.u, time[1], second_layer.x[0,:], second_layer.y[:,0])

dx = first_layer.Delta_x()
dy = first_layer.Delta_y()

K = 2

R_x = K*dt/dx**2
R_y = K*dt/dy**2

a = -0.5 * R_x
b = -0.5 * R_y
c = 1 + R_x + R_y

t_cur = 0.0625

while t_cur < t_max:

    t_cur += dt

    # Initialize A and B matrix using the second layer
    A_mat, B_mat = initialize_matrix(second_layer.u,N-2,a,b,c)

    b_mat = get_b_mat(first_layer.u, N-2)
    updated_B_mat = B_mat + b_mat

    b_vec = updated_B_mat.diagonal()

    x_vec, info = LA.bicgstab(A_mat,b_vec,tol=0.5e-12)

    if info==0:
        # unpack x_vec into u
        for j in range(1, second_layer.Nj-1):
            for i in range(1, second_layer.Ni-1):
                k = (i-1) + (second_layer.Ni-2)*(j-1)
                second_layer.u[j,i]=x_vec[k]

    if t_cur < t_max:
        # Now, second layer is solved
        # Assign it as the new first layer
        first_layer.u = second_layer.u

        # Reset the second layer
        second_layer = reset_layer(second_layer, t_cur)
    else:
        break


if not quiet:
    plot_solution(second_layer.x, second_layer.y, second_layer.u, t_cur)

