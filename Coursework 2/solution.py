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

# for i, level in enumerate(levels):
#     print(np.round(levels[level],3))

# Assemble b matrix
first_layer = levels[0]
print(np.round(first_layer,3))

inner_domain = first_layer[1:-1, 1:-1]

inner_domain_size = len(first_layer) - 2
B = np.zeros((inner_domain_size**2,inner_domain_size**2))


# for j, row in enumerate(inner_domain):
#     for i, col in enumerate(row):
#         k = i + (nj -2) * j

#         # For checking corners
#         top_left_corner = (0,0)
#         top_right_corner = ((len(row) - 1),0)
#         bottom_left_corner = (0,(len(row) - 1))
#         bottom_right_corner = ((len(row) - 1),(len(row) - 1))

#         corner = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

#         # Top left corner
#         if (i,j) == top_left_corner:
#             # Get north
#             north = first_layer[0,1]
#             # Get west
#             west = first_layer[1,0]
#             B[k,k] = 100

#         # Top right corner
#         if (i,j) == top_right_corner:
#             # Get north
#             north = first_layer[0,-2]
#             # Get east
#             east = first_layer[1,-1]
#             B[k,k] = 100

#         # Bottom left corner
#         if (i,j) == bottom_left_corner:
#             # Get west
#             west = first_layer[-2,0]
#             # Get south
#             south = first_layer[-1,1]
#             B[k,k] = 100

#         # Bottom right corner
#         if (i,j) == bottom_right_corner:
#             # Get east
#             east = first_layer[-2,-1]
#             # Get south
#             south = first_layer[-1,-2]
#             B[k,k] = 100
        

#         # Checks the side
#         elif i == 0 and (i,j) not in corner:
#             # Get west 
#             west = first_layer[j+1,0]

#         elif i == (len(row) - 1) and (i,j) not in corner:
#             # Get east
#             east = first_layer[j+1,-1]
        
#         elif j == 0  and (i,j) not in corner:
#             # Get north
#             north = first_layer[0,i+1]
#             print(north)

#         elif j == (len(row) - 1)  and (i,j) not in corner:
#             # Get south
#             south = first_layer[-1,i+1]

inner_size = ni - 2
A_mat = np.zeros(((inner_size)**2, (inner_size)**2))

for j in range(0, inner_size):
    for i in range(0, inner_size):

        top_left_corner = (0,0)
        top_right_corner = ((inner_size - 1),0)
        bottom_left_corner = (0,(inner_size - 1))
        bottom_right_corner = ((inner_size - 1),(inner_size - 1))

        corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

        k = i + (ni - 2) * j
        A_mat[k,k]=1

        # Left boundary
        # if i>1:
        #     A_mat[k,k-1] += 2
        # Check if i is at the middle?
        if i > 0 and i < inner_size - 1:
            A_mat[k,k+1] = 2
            A_mat[k,k-1] = 2

        # Check if it is at the leftmost?
        if i == 0 and (i,j) not in corners:
            A_mat[k,k+1] = 2
            A_mat[k,k+(ni-2)] = 3
            A_mat[k,k-(ni-2)] = 3

        # Check if it is the rightmost?
        if i == inner_size - 1 and (i,j) not in corners:
            A_mat[k, k-1] = 2
            A_mat[k,k+(ni-2)] = 3
            A_mat[k,k-(ni-2)] = 3
            # TODO(Khalid): k + 1 is on the right side so this will be on the b matrix 

         # Check if i is at the middle (in the y-direction)?
        if j > 0 and j < inner_size - 1:
            A_mat[k,k+(ni-2)] = 3
            A_mat[k,k-(ni-2)] = 3

        # Check if it is at the topmost?
        if j == 0 and (i,j) not in corners:
            # Set the south unknown condition
            A_mat[k,k+1] = 2
            A_mat[k,k-1] = 2
            A_mat[k,k+(ni-2)] = 3
            # TODO: k - 3 is the known north condition

        if j == inner_size - 1 and (i,j) not in corners:
            # Set the north unknown condition
            A_mat[k,k+1] = 2
            A_mat[k,k-1] = 2
            A_mat[k,k-(ni-2)] = 3
            # TODO: k + 3 is the known south condition

        # # Check if it is the rightmost?
        # if i == inner_size - 1:
        #     A_mat[k, k-1] = 2
        #     # TODO(Khalid): k + 1 is on the right side so this will be on the b matrix 

        # else:
        #     b_vec[k] += -R_x*grid.u[j,i-1]

        # # Right boundary
        # if i<grid.Ni-2:
        #     A_mat[k,k+1] +=R_x
        # elif (grid.BC[1] == grid.NEUMANN_BC):
        #     A_mat[k,k-1] += R_x
        # else:
        #     b_vec[k] += -R_x*grid.u[j,i+1]
        
        # # Bottom boundary
        # if j>1:
        #     A_mat[k,k-(grid.Ni-2)] += R_y
        # elif (grid.BC[3]== grid.NEUMANN_BC):
        #     A_mat[k,k+(grid.Ni-2)] += R_y
        # else:
        #     b_vec[k] += -R_y*grid.u[j-1,i]

        # # South boundary
        # if j<grid.Nj-2:
        #     A_mat[k,k+(grid.Ni-2)] += R_y
        # elif (grid.BC[2]== grid.NEUMANN_BC):
        #     A_mat[k,k-(grid.Ni-2)] += R_y
        # else:
        #     b_vec[k] += -R_y*grid.u[j+1,i]

print(A_mat)

# mat_b = initialize_b_matrix(inner_domain)


