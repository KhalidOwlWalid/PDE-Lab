
import numpy as np
from sympy import N
from grid import Grid
import matplotlib.pyplot as plt
from numpy.linalg import solve
import scipy.sparse.linalg as LA
import time
import scipy.sparse as sps
from mpl_toolkits import mplot3d
from refinement_analysis import refinement_analysis
import scipy.integrate as integrate


class CrankNicholson(Grid):

    def __init__(self):
        
        self.K = 2
        self.t_max = 2

    def set_boundary_condition(self,u_grid, t,x,y, K=2):
        # Top
        u_grid[0,:] = 0
        # Bottom
        u_grid[-1,:] = 0
        # Left
        u_grid[:,0] = 0.5 * np.cos(y) * np.exp(-K*t)
        # Right 
        u_grid[:,-1] = t * np.cos(y)

    def set_initial_condition(self,first_layer,x,y):
        first_layer[1:-1,1:-1] = 0.5 * np.cos(x) * np.cos(y)
        return first_layer

    def plot_solution(self,ax, x,u,y=None,plot3d=False,plot1d=False):

        if plot3d:
            ax = plt.axes(projection='3d')
            ax.contour3D(x, y, u, 50, cmap='binary')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')

        if plot1d:
            ax.scatter(x,u)

        else:
            cmap = plt.get_cmap('jet')
            cf = ax.contourf(x,y, u, cmap=cmap, levels = 21)
            fig.colorbar(cf, ax=ax)

    def spy_matrix(self,matrix, inner_grid_size, quiet=True):

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

    def initialize_matrix(self,ni,nj,cur_layer,inner_grid_size,a,b,c):
        
        N = (inner_grid_size)**2
        A_mat = sps.lil_matrix((N, N), dtype=np.float64)
        # A_mat = np.zeros(((inner_grid_size)**2, (inner_grid_size)**2))

        # Remember that B_mat should be converted to a 1 x inner_grid_size matrix 
        # TODO(Khalid): Use B_mat.diagonal() to extract the B_matrix
        B_mat = np.zeros((inner_grid_size**2,inner_grid_size**2))

        for j in range(0, inner_grid_size):
            for i in range(0, inner_grid_size):

                k = i + (ni - 2) * j
                A_mat[k,k] += c

                top_left_corner = (0,0)
                top_right_corner = ((inner_grid_size - 1),0)
                bottom_left_corner = (0,(inner_grid_size - 1))
                bottom_right_corner = ((inner_grid_size - 1,inner_grid_size - 1))

                corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

                # Check if i is at the middle?
                if i > 0 and i < inner_grid_size - 1:
                    #  Set the coefficient of unknown west value
                    A_mat[k,k+1] += b
                    A_mat[k,k-1] += b

                # Check if it is at the leftmost?
                elif i == 0:
                    # Set the coefficient of unknown west value
                    A_mat[k,k+1] += b

                    if (i,j) in corners:
                        if k == 0:
                            # Get the north and west BC
                            B_mat[k,k] += -a * cur_layer[0,1] - b * cur_layer[1,0] 
                            
                        else:
                            # Get the south and west BC
                            B_mat[k,k] += -a * cur_layer[-1,1] - b * cur_layer[-2,0]
                    else:
                        B_mat[k,k] += -b * cur_layer[j+1,0]

                # Check if it is the rightmost?
                elif i == inner_grid_size - 1:
                    #  Set the coefficient of unknown east value
                    A_mat[k, k-1] += b

                    if (i,j) in corners:
                        # Get the north and east boundary condition and pass it to B_mat
                        # Top right corner
                        if k == inner_grid_size - 1:
                            B_mat[k,k] += -a * cur_layer[0,-2] -b * cur_layer[1,-1]
                        # Bottom right corner
                        else:
                            # Get south and east boundary
                            B_mat[k,k] += -a * cur_layer[-1,-2] - b * cur_layer[-2,-1]
                    else:
                        B_mat[k,k] += -b * cur_layer[j+1,-1]

                # Check if j is at the middle (in the y-direction)?
                if j > 0 and j < inner_grid_size - 1:
                    #  Set the coefficient of unknown north value
                    A_mat[k,k+(ni-2)] += a
                    A_mat[k,k-(ni-2)] += a

                # Check if it is at the topmost?
                elif j == 0:
                    #  Set the coefficient of unknown south value
                    A_mat[k,k+(ni-2)] += a
                    
                    if (i,j) not in corners:
                        # Get the north known boundary condition
                        B_mat[k,k] += -a * cur_layer[0,i+1]

                # Check if it is at the bottom most?
                elif j == inner_grid_size - 1:
                    #  Set the coefficient of unknown south value
                    A_mat[k,k-(ni-2)] += a
                    
                    if (i,j) not in corners:
                        # Get the south known boundary condition
                        B_mat[k,k] += -a * cur_layer[-1,i+1]

        # In later stage, please return B_mat.diagonal()
        return A_mat, B_mat

    def get_b_mat(self,prev_layer,inner_grid_size,r_x, r_y):
        
        b_mat = np.zeros((inner_grid_size**2, inner_grid_size**2))

        for j in range(1, inner_grid_size+1):
            for i in range(1, inner_grid_size+1):
                # Get all values
                # Format : b_mat[k,k] += origin + north + south + east + west
                origin = prev_layer[j,i]
                north = prev_layer[j-1,i]
                south = prev_layer[j+1,i]
                east = prev_layer[j,i+1]
                west = prev_layer[j,i-1] 

                sum = origin + r_x/2 * (east - 2 * origin + west) + r_y/2 * (south - 2 * origin + north)
                # sum = prev_layer[j,i] + prev_layer[j-1,i] + prev_layer[j+1,i] + prev_layer[j,i+1]  + prev_layer[j,i-1] 

                m = i - 1
                n = j - 1

                k = m + inner_grid_size * n

                b_mat[k,k] += sum

        return b_mat

    def reset_layer(self,ni,nj,new_cur_layer,t):
        new_cur_layer = Grid(ni,nj)
        new_cur_layer.set_origin(0,-np.pi/2)
        new_cur_layer.set_extent(np.pi/2,np.pi/2)
        new_cur_layer.generate()

        # Set the boundary condition
        self.set_boundary_condition(new_cur_layer.u, t, new_cur_layer.x[0,:], new_cur_layer.y[:,0])

        return new_cur_layer

    # Courant number of 0.95 is a safe choice
    def main(self,N,t_max,v=0.95, quiet=False, plot3d=False):

        clock_start = time.time()

        assert N % 2 != 0, "N should be odd"
        assert v > 0, "Courant number,v needs to be larger than 0"

        
        ni = N
        nj = N

        t = 0

        # Initialize layers
        first_layer = Grid(ni,nj)
        first_layer.set_origin(0,-np.pi/2)
        first_layer.set_extent(np.pi/2,np.pi/2)
        first_layer.generate()

        second_layer = Grid(ni,nj)
        second_layer.set_origin(0,-np.pi/2)
        second_layer.set_extent(np.pi/2,np.pi/2)
        second_layer.generate()

        dt = v * min(first_layer.Delta_x()**2/(2*first_layer.kappa),
                    first_layer.Delta_y()**2/(2*first_layer.kappa))

        print(f'The dt for {N} is {dt}')

        # Set the initial condition of the first layer
        self.set_boundary_condition(first_layer.u, 0, first_layer.x[0,:], first_layer.y[:,0])
        self.set_boundary_condition(second_layer.u, t + dt, second_layer.x[0,:], second_layer.y[:,0])
        self.set_initial_condition(first_layer.u, first_layer.x[1:-1,1:-1], first_layer.y[1:-1,1:-1])

        # Calculate coefficients
        dx = first_layer.Delta_x()
        dy = first_layer.Delta_y()

        R_x = self.K*dt/dx**2
        R_y = self.K*dt/dy**2

        a = -0.5*R_y
        b = -0.5*R_x
        c = 1 + R_x + R_y

        # Initialize A and B matrix using the second layer
        A_mat, BC = self.initialize_matrix(ni,nj,second_layer.u,N-2,a,b,c)
        
        # previous (n layer)
        b_mat = self.get_b_mat(first_layer.u, N-2, R_x, R_y)
        
        updated_B_mat = BC + b_mat
        b_vec = updated_B_mat.diagonal()

        x_vec, info = LA.bicgstab(A_mat,b_vec,tol=0.5e-12)

        if info==0:
            # unpack x_vec into u
            for j in range(1, second_layer.Nj-1):
                for i in range(1, second_layer.Ni-1):
                    k = (i-1) + (second_layer.Ni-2)*(j-1)
                    second_layer.u[j,i]=x_vec[k]

        steps = int(self.t_max/dt)
        t_end = dt * steps

        if not quiet:
            for i, t in enumerate(np.arange(dt,t_end + dt,dt)):
                # Update the second layer as the first layer now
                first_layer.u = second_layer.u
                second_layer = self.reset_layer(ni,nj,second_layer,t)


                A_mat, B_mat = self.initialize_matrix(ni,nj,second_layer.u,N-2,a,b,c)
                b_mat = self.get_b_mat(first_layer.u, N-2, R_x, R_y)
                updated_B_mat = B_mat + b_mat

                b_vec = updated_B_mat.diagonal()

                ilu = LA.spilu(A_mat.tocsc(), drop_tol=1e-6, fill_factor=100)
                M_mat = LA.LinearOperator(A_mat.shape, ilu.solve)
                x_vec, info = LA.bicgstab(A_mat,b_vec,atol=5e-12,M=M_mat)

                if info==0:
                    # unpack x_vec into u
                    for j in range(1, second_layer.Nj-1):
                        for i in range(1, second_layer.Ni-1):
                            k = (i-1) + (second_layer.Ni-2)*(j-1)
                            second_layer.u[j,i]=x_vec[k]

        clock_end = time.time()

        time_taken = clock_end - clock_start
            
        return second_layer, second_layer.x, second_layer.y, second_layer.u, np.round(time_taken,2), dx

    def analytical_solution(self,x,repetition=5):

        summ = 0
        for n in range (1, repetition):
            value = 1/(2*self.K*np.pi)*((-1)**n/(n**3)*(1-np.exp(-4*self.K*n**2*self.t_max))*np.sin(2*n*x))
            summ += value
        summ += 2*x*self.t_max/np.pi + 0.5 * np.cos(x) * np.exp(-self.K*self.t_max)

        return summ

    def grid_converged_solution(self,N,x_grid,u_grid):

        mid = int((N+1)/2)
        
        numerical_soln = u_grid[mid-1]
        analytical_soln = self.analytical_solution(x_grid[N-1],10)

        diff = abs(numerical_soln - analytical_soln)
        accuracy = sum(diff)/len(diff)

        return accuracy

    def print_grid(self,grid,sig_fig=3):
        print(np.round(grid,sig_fig))

    def integrate_u_dx(self,mesh,x0,x1,y):
        '''Calculate U=\int_{1.0}^{1.95)u(x,0.25) dx  using the
        u value stored on the grid and simpsons rule'''
        
        # find the left and right grid points
        i0,j = mesh.find((x0,y))
        i1,j = mesh.find((x1,y))

        # add 1 to i1 as we are going to use it as an upper 
        # bound forarray slicing
        i1 = i1 +1
        
        # integrate
        return integrate.simps(mesh.u[j,i0:i1],mesh.x[j,i0:i1])

if __name__ == "__main__":

    crank_nicholson = CrankNicholson()

    N = [5,21,31]
    
    result = {}

    dx_val = []
    u_val = []
    run_time = []
    n_pts =[]

    # for grid_index in range(4,1,-1):
    for grid_index in [7,15,27]:

        ni = 5*2**grid_index + 1
        nj = 5*2**grid_index + 1
        n_pts.append(ni*nj)

        grid_size = ni

        print(f"Running for grid size {grid_size}...")
        mid = int((grid_size+1)/2) 
        soln_grid, x_grid, y_grid, u_grid, time_taken, dx = crank_nicholson.main(N=grid_size,t_max=2,v=0.95,quiet=False,plot3d=False)   

        converged_soln = crank_nicholson.grid_converged_solution(grid_size,x_grid,u_grid)

        # Store the solution
        result[grid_size] = [x_grid[mid -1], u_grid[mid - 1], time_taken, converged_soln] 

        print(f"Time taken to solve {time_taken}")   

        dx_val.append(dx)
        u_val.append(crank_nicholson.integrate_u_dx(soln_grid,1.0,1.95,0.25))
        run_time.append(time_taken)

        print('Integrated value is ',u_val[-1],'\n')

from refinement_analysis import refinement_analysis
# lets to the refinement analysis
analysis = refinement_analysis(dx_val,u_val)
print(dx_val)
print(u_val)
analysis.report(r'\int_{75}^{125}h(x,25) dx')
analysis.plot(True,r'$\int_{75}^{125}h(x,25) dx$')
# fig1, ax1 = plt.subplots()

# for n, grid_size in enumerate(N):
#     ax1.scatter(result[grid_size][0],result[grid_size][1], label=f'Grid size: {grid_size}')

# analytical_soln = crank_nicholson.analytical_solution(result[5][0])

# ax1.plot(result[5][0],analytical_soln, color='r', label='Analytical solution')
# plt.legend()
# plt.show()
