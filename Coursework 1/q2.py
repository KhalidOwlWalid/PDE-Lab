import numpy as np
import scipy.sparse.linalg as LA
import scipy.sparse as sps
import matplotlib.pyplot as plt
import time

class Grid:
    '''Class defining a 2D computational grid for
    a polar cartesian grid.  We assume that theta
    goes from 0 to 2\pi and has periodic boundaires
    in the theta direction.'''
    
    def __init__(self,ni,nj):
        # set up information about the grid
        self.origin = 0.0 # unit circle  
        self.extent = 1.0
        self.Ni = ni # grid points in i direction
        self.Nj = nj # grid points in j direction
        
        # initialse x,y and u arrays
        self.u = np.zeros((nj, ni))
        self.r = np.zeros((nj, ni))
        self.theta = np.zeros((nj, ni))

    def set_origin(self, r0):
        self.origin = r0
    
    def set_extent(self,r1):
        self.extent = r1
        
    def generate(self,Quiet=True):
        '''generate a uniformly spaced grid covering the domain from the
        origin to the extent.  We are going to do this using linspace from
        numpy to create lists of x and y ordinates and then the meshgrid
        function to turn these into 2D arrays of grid point ordinates.'''
        
        theta = np.linspace(0.0, 2*np.pi, self.Ni+1)
        theta_ord = np.delete(theta,[-1])
        r_ord = np.linspace(self.origin, self.extent, self.Nj)
        self.r, self.theta = np.meshgrid(r_ord, theta_ord)
        if not Quiet:
            print(self)

    def Delta_r(self):
        # calculate delta x
        return self.r[0,1]-self.r[0,0]
    
    def Delta_theta(self):
        # calculate delta y
        return self.theta[1,0]-self.theta[0,0]
    
    def find(self,point):
        '''find the i and j ordinates of the grid cell which contains 
        the point (theta,r).  To do this we calculate the distance from
        the point to the origin in the x and y directions and then
        divide this by delta x and delta y.  The resulting real ordinates
        are converted to indices using the int() function.'''
        grid_x = (point[0] - self.origin[0])/self.Delta_theta()
        grid_y = (point[1] - self.origin[1])/self.Delta_r()
        return int(grid_x), int(grid_y)
    
    def plot(self,title):
        '''produce a contour plot of u(theta,r)'''
        # convert polar to Cartesian coordinates
        x = self.r*np.cos(self.theta)
        y = self.r*np.sin(self.theta)
        
        # produce the plot
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('jet')
        ax.axis('equal')
        ax.set_aspect('equal', 'box')
        cf = ax.contourf(x,y,self.u,cmap=cmap,levels = 21)
        fig.colorbar(cf, ax=ax)
        ax.set_title(f'{title} ({self.Ni} x {self.Nj} grid)')
        return plt
    
    def __str__(self):
        # describe the object when asked to print it
        return 'Uniform {}x{} polar grid from r = {} to {}.'.format(self.Ni, self.Nj, self.origin, self.extent)

#Set up the coursework 
def coursework_one(ni,nj):
    # set up a mesh
    mesh = Grid(ni,nj)
    mesh.set_origin(1.0)
    mesh.set_extent(3.0)
    mesh.generate()
    return mesh

# lets set up the test problem
test = coursework_one(41,41)
print("cell run")

def PolarLaplaceSolver(grid,tol=0.5e-7):
    '''Solve the two dimensional laplace equation on a polar coordinates
    using the bi-conjugate gradient stabilised matrix solver (BiCGStab).  
    This function assembles the coeficient matrix A and the RHS vector b
    taking account of the boundary conditions specified in the question.
    It should call the  BiCGStab solver from scipy.  The value tol is p
    assed to BiCGStab routine The solution vector x is then unpacked 
    into grid.u. It returns the info value from BiCGStab if this is 
    zero everything worked.'''
    
    # Create the A matrix using the lil format and the b vector
    # as numpy vector.
    N = (grid.Nj-2)*(grid.Ni-2)
    A_mat = sps.lil_matrix((N, N), dtype=np.float64)
    b_vec = np.zeros(N, dtype=np.float64)
    
    dr = grid.Delta_r()
    dtheta = grid.Delta_theta()
    
    N = (grid.Nj-2)*(grid.Ni-2)

    # calculate the coefficients
#     beta = grid.Delta_r()/grid.Delta_theta()
#     beta_sq = beta**2
#     R_x = - 1/(2*(1+beta_sq))
#     R_y = beta_sq * R_x

    # -----
    for j in range(1, grid.Ni-1):
        for i in range(1, grid.Ni-1):
            
            
            k = (i-1) + (grid.Ni-2)*(j-1)
            
            r = grid.r[j,i]
            theta = grid.theta[j,i]
            
            R_o = - 2 * (2 * r * dtheta ** 2 + dr ** 2)
            R_n = dr**2 * dtheta**2 * r + 0.5 * dr * dtheta**2
            R_s = dr**2 * dtheta**2 * r - 0.5 * dr * dtheta**2 
            R_ew = dr**2

            A_mat[k, k] = R_o

            # Left boundary (DIRICHLET)
            if i > 1:
                A_mat[k, k - 1] = 2 * r * dtheta ** 2 - 0.5 * r * dr * dtheta ** 2
            else:
                b_vec[k] += - (2 * r * dtheta ** 2 - 0.5 * r * dr * dtheta ** 2) * grid.u[j, i - 1]



            # Right boundary (NEUMANN)
            if i < grid.Ni - 2:
                A_mat[k, k + 1] = 2 * r * dtheta ** 2 + 0.5 * r * dr * dtheta ** 2
            else:
                A_mat[k, k - 1] += 2 * r * dtheta ** 2 + 0.5 * r * dr * dtheta ** 2
                if theta >= np.pi:
                    b_vec[k] += - 2 * dr * (2 * r * dtheta ** 2 + 0.5 * r * dr * dtheta ** 2)



            # Top boundary (PERIODIC)
            if j < grid.Nj - 2:
                A_mat[k, k + (grid.Ni - 2)] = dr ** 2
            else:
                A_mat[k,k+(grid.Ni-2)-N]=dr**2


            # Bottom boundary (PERIODIC)
            if j > 1:
                A_mat[k, k -(grid.Ni - 2)] = dr ** 2
            else:
                A_mat[k,k-(grid.Ni-2)]=dr**2
                
    # -----
    
    # call bicgstab
    x_vec, info = LA.bicgstab(A_mat,b_vec,tol=tol)
    
    if info==0:
        # unpack x_vec into u
        for j in range(1, grid.Nj-1):
            for i in range(1, grid.Ni-1):
                k = (i-1) + (grid.Ni-2)*(j-1)
                grid.u[j,i]=x_vec[k]
    
    print(x_vec)
    return info

# Test the solution on the 40x40 grid
info = PolarLaplaceSolver(test)

if info==0:
    plt = test.plot('Coursework 1')
    plt.show()
else:
    print('Error code ',info,' returned by BiCGStab')