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

    for j in range(1, grid.Ni-1):
        for i in range(1, grid.Ni-1):
            
            
            k = (i-1) + (grid.Ni-2)*(j-1)
            
            r = grid.r[j,i]
            theta = grid.theta[j,i]
            
            alpha = 2 * r * dtheta**2
            beta = 0.5 * r * dr * dtheta**2
            gamma = dr**2
            
            R_o = -2 * ( alpha + gamma)
            R_n = alpha + beta
            R_s = alpha - beta
            R_ew = gamma

            A_mat[k, k] = R_o

            # Left boundary (DIRICHLET)
            # South
            if i > 1:
                A_mat[k, k - 1] = R_s
            else:
                b_vec[k] += - R_s * grid.u[j, i - 1]



            # Right boundary (NEUMANN)
            # North
            if i < grid.Ni - 2:
                A_mat[k, k + 1] = R_n
            else:
                A_mat[k, k - 1] += R_n
                if theta >= np.pi:
                    b_vec[k] += - 2 * dr * R_n



            # Top boundary (PERIODIC)
            # West
            if j < grid.Nj - 2:
                A_mat[k, k + (grid.Ni - 2)] = R_ew
            else:
                A_mat[k,k+(grid.Ni-2)-N]= R_ew


            # Bottom boundary (PERIODIC)
            # East
            if j > 1:
                A_mat[k, k -(grid.Ni - 2)] = R_ew
            else:
                A_mat[k,k-(grid.Ni-2) + N]= R_ew
                
    plt.spy(A_mat, markersize=2, aspect='equal')
    plt.show()
    # -----
    
    # call bicgstab
    x_vec, info = LA.bicgstab(A_mat,b_vec,tol=tol)
    
    if info==0:
        # unpack x_vec into u
        for j in range(1, grid.Nj-1):
            for i in range(1, grid.Ni-1):
                k = (i-1) + (grid.Ni-2)*(j-1)
                grid.u[j,i]=x_vec[k]
    
    return info