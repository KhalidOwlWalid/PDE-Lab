import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm

class Grid:
    '''Class defining a 2D computational grid for
    the solution of the Laplace equation.  '''
    
    def __init__(self,ni,nj):
        # set up information about the grid
        self.origin = (0.0, 0.0)  # bottom left
        self.extent = (1.0, 1.0)  # top right
        self.Ni = ni # grid points in i direction
        self.Nj = nj # grid points in j direction
        
        # initialse x,y and u arrays
        self.u = np.zeros((nj, ni))
        self.x = np.zeros((nj, ni))
        self.y = np.zeros((nj, ni))

    def set_origin(self,x0,y0):
        '''Specity the location of the bottom left corner of the grid'''
        self.origin = (x0, y0)
    
    def set_extent(self,x1,y1):
        '''Specity the width and height of the grid'''
        self.extent = (x1, y1)
        
    def generate(self,Quiet=True):
        '''generate a uniformly spaced grid covering the domain from the
        origin to the extent.  We are going to do this using linspace from
        numpy to create lists of x and y ordinates and then the meshgrid
        function to turn these into 2D arrays of grid point ordinates.'''
        x_ord = np.linspace(self.origin[0], self.extent[0], self.Ni)
        y_ord = np.linspace(self.origin[1], self.extent[1], self.Nj)
        self.x, self.y = np.meshgrid(x_ord,y_ord)
        if not Quiet:
            print(self)

    def Delta_x(self):
        # calculate delta x
        return self.x[0,1]-self.x[0,0]
    
    def Delta_y(self):
        # calculate delta y
        return self.y[1,0]-self.y[0,0]
    
    def find(self,point):
        '''find the i and j ordinates of the grid cell which contains 
        the point (x,y).  To do this we calculate the distance from
        the point to the origin in the x and y directions and then
        divide this by delta x and delta y.  The resulting real ordinates
        are converted to indices using the int() function.'''
        grid_x = (point[0] - self.origin[0])/self.Delta_x()
        grid_y = (point[1] - self.origin[1])/self.Delta_y()
        return int(grid_x), int(grid_y)
    
    def __str__(self):
        # describe the object when asked to print it
        return 'Uniform {}x{} grid from {} to {}.'.format(self.Ni, self.Nj, self.origin, self.extent)
## Write code here to define a function setting up the problem, and a function implementing the Jacobi itteration.
## Both functions should opperate on an object called mesh which is of the Grid class.  The solution should be
## stored in mesh.u.  

def lab2(ni,nj):
    # set up a mesh
    mesh = Grid(ni,nj)
    mesh.set_extent(1.0,4.0)
    mesh.generate()
    # now the RHS boundary condition
    mesh.u[-1,:]=np.sin(np.pi*mesh.x[-1,:])**5
    return mesh

## Write code here to solve the PDE and plot the solution using an appropriate sized grid.  
# lets test it
mesh = lab2(100,100)

def Jacobi(mesh,tol=0.5e-7,maxit=10000):
    '''Jacobi itteration applied to the grid stored in mesh.  
    We will continue itterating until the difference between
    u^{n+1} and u^n is less than tol. We will also stop if 
    we have done more than maxit itterations.
    
    The solution stored in the mesh.u variable is updated'''
    
    # calculate the coefficients
    beta = mesh.Delta_x()/mesh.Delta_y()
    beta_sq = beta**2
    C_beta = 1/(2*(1+beta_sq))
    
    # initialise u_new 
    u_new = mesh.u.copy()
    
    # itteration
    for it in range(maxit):

        u_new[1:-1,1:-1] = C_beta*(mesh.u[1:-1,:-2]+mesh.u[1:-1,2:]+
                                  beta_sq*(mesh.u[:-2,1:-1]+mesh.u[2:,1:-1]))
        
        # compute the difference between the new and old solutions
        err = np.max(abs(mesh.u-u_new))/np.max(mesh.u)
        
        # update the solution
        mesh.u = np.copy(u_new)
        
        # converged?
        if err < tol:
            break
    
    if it+1 == maxit:
        print('Jacobi itteration failed to converge, error = {}'.format(err))
    
    return it+1, err # return the number of itterations and the final residual

iterations, error = Jacobi(mesh)
print(iterations, error)

# plot the solution
fig, ax1 = plt.subplots()
cmap = plt.get_cmap('PiYG')
cf = ax1.contourf(mesh.x, mesh.y, mesh.u,cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title(f'Example 9.29 ({mesh.Ni} x {mesh.Nj} grid)')

plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(mesh.x,mesh.y,mesh.u,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


plt.show()