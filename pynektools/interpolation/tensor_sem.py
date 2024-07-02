import numpy as np

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from math import pi
from scipy.special import legendre

class element_interpolator_c():
    def __init__(self, n, modal_search = True, max_pts = 1, max_elems = 1):
    
        # Order of the element
        self.n = n
        
        # Type of basis to use
        self.modal_search = modal_search

        # Maximun of points to interpolate at the same time
        self.max_pts = max_pts

        # Maximun number of elements to interpolate at the same time
        self.max_elemes = max_elems

        # Get reference element (On the CPU)
        get_reference_element(self)

        # Get reference element
        get_basis_transformation_matrices(self)
        
        # Allocate relevant arrays
        ## The xyz coordinates are invariant
        self.xj = np.zeros((max_pts,1, 1, 1))
        self.yj = np.zeros((max_pts,1, 1, 1))
        self.zj = np.zeros((max_pts,1, 1, 1))
        ## The rst coordinates depend on the point and on the element
        self.rj = np.zeros((max_pts,max_elems, 1, 1))
        self.sj = np.zeros((max_pts,max_elems, 1, 1))
        self.tj = np.zeros((max_pts,max_elems, 1, 1))
        self.rstj = np.zeros((max_pts, max_elems, 3,1))
        self.eps_rst = np.ones((max_pts, max_elems, 3,1))
        self.jac = np.zeros((max_pts, max_elems, 3,3))
        self.iterations = 0
        self.field_e = np.zeros_like(self.x_e)
        self.point_inside_element = False
         
        return

    def project_element_into_basis(self, x_e, y_e, z_e, apply_1d_ops = True):
        
        npoints = x_e.shape[0]
        nelems = x_e.shape[1]
        n = x_e.shape[2]*x_e.shape[3]*x_e.shape[4]

        # Assing the inputs to proper formats
        self.x_e[:npoints,:nelems,:,:] = x_e.reshape(npoints, nelems, n, 1)[:,:,:,:]
        self.y_e[:npoints,:nelems,:,:] = y_e.reshape(npoints, nelems, n, 1)[:,:,:,:]
        self.z_e[:npoints,:nelems,:,:] = z_e.reshape(npoints, nelems, n, 1)[:,:,:,:]

        if self.modal_search:
            # Get the modal representation
            if not apply_1d_ops:    
                
                self.x_e_hat = np.einsum('ijkl,ijlm->ijkm', self.v_inv, self.x_e[:npoints,:nelems,:,:])
                self.y_e_hat = np.einsum('ijkl,ijlm->ijkm', self.v_inv, self.y_e[:npoints,:nelems,:,:])
                self.z_e_hat = np.einsum('ijkl,ijlm->ijkm', self.v_inv, self.z_e[:npoints,:nelems,:,:])
            
            else:
                
                # Keep in mind, the operators are already transposed here.
                self.x_e_hat = apply_operators_3d(self.v1d_inv, self.v1d_inv, self.v1d_inv, self.x_e[:npoints,:nelems,:,:])
                self.y_e_hat = apply_operators_3d(self.v1d_inv, self.v1d_inv, self.v1d_inv, self.y_e[:npoints,:nelems,:,:])
                self.z_e_hat = apply_operators_3d(self.v1d_inv, self.v1d_inv, self.v1d_inv, self.z_e[:npoints,:nelems,:,:]) 
        else:
            raise RuntimeError("Non-modal search is not implemented with tensor support")

        return
 
    def get_xyz_from_rst(self, rj, sj, tj, apply_1d_ops = True):

        '''
        This function calculates the xyz coordinates from the given rst coordinates for points
        in the elements that were projected into xhat, yhat, zhat.
        '''

        npoints = rj.shape[0]
        nelems = self.x_e_hat.shape[1]
        n = self.n

        self.rj[:npoints,:nelems,:,:] = rj[:,:,:,:]
        self.sj[:npoints,:nelems,:,:] = sj[:,:,:,:]
        self.tj[:npoints,:nelems,:,:] = tj[:,:,:,:]

        # Find the basis for each coordinate separately
        if self.modal_search:
            # If modal search, the basis is legendre
            ortho_basis_rj = legendre_basis_at_xtest(n, self.rj[:npoints,:nelems,:,:])
            ortho_basis_sj = legendre_basis_at_xtest(n, self.sj[:npoints,:nelems,:,:])
            ortho_basis_tj = legendre_basis_at_xtest(n, self.tj[:npoints,:nelems,:,:])

            ortho_basis_prm_rj = legendre_basis_derivative_at_xtest(ortho_basis_rj, self.rj[:npoints,:nelems,:,:])
            ortho_basis_prm_sj = legendre_basis_derivative_at_xtest(ortho_basis_sj, self.sj[:npoints,:nelems,:,:])
            ortho_basis_prm_tj = legendre_basis_derivative_at_xtest(ortho_basis_tj, self.tj[:npoints,:nelems,:,:])

        else:
            raise RuntimeError("Non-modal search is not implemented with tensor support")

            
        if not apply_1d_ops:
            raise RuntimeError("Only worrking by applying 1d operators")

        elif apply_1d_ops:
            
            x = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2), ortho_basis_sj.transpose(0,1,3,2), ortho_basis_tj.transpose(0,1,3,2), self.x_e_hat)   
            y = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2), ortho_basis_sj.transpose(0,1,3,2), ortho_basis_tj.transpose(0,1,3,2), self.y_e_hat)
            z = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2), ortho_basis_sj.transpose(0,1,3,2), ortho_basis_tj.transpose(0,1,3,2), self.z_e_hat)

            self.jac[:npoints, :nelems, 0, 0] = apply_operators_3d(ortho_basis_prm_rj.transpose(0,1,3,2), ortho_basis_sj.transpose(0,1,3,2)    , ortho_basis_tj.transpose(0,1,3,2), self.x_e_hat)[:,:,0,0]   
            self.jac[:npoints, :nelems, 0, 1] = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2)    , ortho_basis_prm_sj.transpose(0,1,3,2), ortho_basis_tj.transpose(0,1,3,2), self.x_e_hat)[:,:,0,0]   
            self.jac[:npoints, :nelems, 0, 2] = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2)    , ortho_basis_sj.transpose(0,1,3,2)    , ortho_basis_prm_tj.transpose(0,1,3,2), self.x_e_hat)[:,:,0,0]    

            self.jac[:npoints, :nelems, 1, 0] = apply_operators_3d(ortho_basis_prm_rj.transpose(0,1,3,2), ortho_basis_sj.transpose(0,1,3,2)    , ortho_basis_tj.transpose(0,1,3,2), self.y_e_hat)[:,:,0,0]    
            self.jac[:npoints, :nelems, 1, 1] = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2)    , ortho_basis_prm_sj.transpose(0,1,3,2), ortho_basis_tj.transpose(0,1,3,2), self.y_e_hat)[:,:,0,0]    
            self.jac[:npoints, :nelems, 1, 2] = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2)    , ortho_basis_sj.transpose(0,1,3,2)    , ortho_basis_prm_tj.transpose(0,1,3,2), self.y_e_hat)[:,:,0,0]    

            self.jac[:npoints, :nelems, 2, 0] = apply_operators_3d(ortho_basis_prm_rj.transpose(0,1,3,2), ortho_basis_sj.transpose(0,1,3,2)    , ortho_basis_tj.transpose(0,1,3,2), self.z_e_hat)[:,:,0,0]    
            self.jac[:npoints, :nelems, 2, 1] = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2)    , ortho_basis_prm_sj.transpose(0,1,3,2), ortho_basis_tj.transpose(0,1,3,2), self.z_e_hat)[:,:,0,0]    
            self.jac[:npoints, :nelems, 2, 2] = apply_operators_3d(ortho_basis_rj.transpose(0,1,3,2)    , ortho_basis_sj.transpose(0,1,3,2)    , ortho_basis_prm_tj.transpose(0,1,3,2), self.z_e_hat)[:,:,0,0]    

        return x, y, z

    def find_rst_from_xyz(self, xj, yj, zj, tol = np.finfo(np.double).eps*10, max_iterations = 1000):

        self.point_inside_element = False
        
        npoints = xj.shape[0]
        nelems = self.x_e_hat.shape[1]
        n = self.n
               
        self.xj[:npoints,:,:,:] = xj[:,:,:,:]
        self.yj[:npoints,:,:,:] = yj[:,:,:,:]
        self.zj[:npoints,:,:,:] = zj[:,:,:,:]

        # Determine the initial conditions
        determine_initial_guess(self, npoints = npoints, nelems = nelems) # This populates self.rj, self.sj, self.tj for 1st iteration

        # Use the newton method to identify the coordinates
        self.iterations = 0
        self.eps_rst[:npoints, :nelems, :,:] = 1
        
        while np.any(np.linalg.norm(self.eps_rst[:npoints, :nelems], axis=(2,3)) > tol) and self.iterations < max_iterations:

            # Update the guess
            self.rstj[:npoints, :nelems, 0,0] = self.rj[:npoints,:nelems,0,0]
            self.rstj[:npoints, :nelems, 1,0] = self.sj[:npoints,:nelems,0,0]
            self.rstj[:npoints, :nelems, 2,0] = self.tj[:npoints,:nelems,0,0]

            # Estimate the xyz values from rst and also the jacobian (it is updated inside self.jac)
            # The elements are determined by the number of x_hats, this is given in the projection function
            # Check that one out if you forget.
            xj_found, yj_found, zj_found = self.get_xyz_from_rst(self.rj[:npoints,:nelems,:,:], self.sj[:npoints,:nelems,:,:], self.tj[:npoints,:nelems,:,:])

            # Find the residuals and the jacobian inverse.
            self.eps_rst[:npoints, :nelems, 0, 0] = (self.xj[:npoints, :nelems, :, :] - xj_found)[:,:,0,0]
            self.eps_rst[:npoints, :nelems, 1, 0] = (self.yj[:npoints, :nelems, :, :] - yj_found)[:,:,0,0]
            self.eps_rst[:npoints, :nelems, 2, 0] = (self.zj[:npoints, :nelems, :, :] - zj_found)[:,:,0,0]
            jac_inv = np.linalg.inv(self.jac[:npoints, :nelems])

            # Find the new guess
            self.rstj[:npoints, :nelems] = self.rstj[:npoints, :nelems] - (0 - np.einsum('ijkl,ijlm->ijkm', jac_inv, self.eps_rst[:npoints,:nelems]))

            # Update the values
            self.rj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 0, 0]
            self.sj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 1, 0]
            self.tj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 2, 0]
            self.iterations += 1

        # Here I am omiting some logic to check if the point is inside the element.
        # This is present in sem.py, so check there for reference how it has been done. 
        
        return self.rj[:npoints, :nelems], self.sj[:npoints, :nelems], self.tj[:npoints, :nelems]

def determine_initial_guess(self, npoints = 1, nelems = 1):
    '''
    Note: Find a way to evaluate if this routine does help. 
    It might be that this is not such a good way of making the guess.
    '''

    if self.modal_search:

        ## Find the closest point for each element
        #diff_x = self.x_e_hat.reshape(npoints, nelems, self.n, self.n, self.n) - self.xj[:npoints,:,:,:].reshape(npoints, 1, 1, 1, 1)
        #diff_y = self.y_e_hat.reshape(npoints, nelems, self.n, self.n, self.n) - self.yj[:npoints,:,:,:].reshape(npoints, 1, 1, 1, 1)
        #diff_z = self.z_e_hat.reshape(npoints, nelems, self.n, self.n, self.n) - self.zj[:npoints,:,:,:].reshape(npoints, 1, 1, 1, 1)

        #distances = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        
        #minim = np.min(distances, axis=(2,3,4)).reshape(npoints, nelems, 1, 1, 1)
        #min_index = np.where(distances == minim)

        #self.rj[min_index[0],min_index[1],0,0] = self.x_gll[min_index[4],0,0,0]
        #self.sj[min_index[0],min_index[1],0,0] = self.x_gll[min_index[3],0,0,0]
        #self.tj[min_index[0],min_index[1],0,0] = self.x_gll[min_index[2],0,0,0]

        self.rj[:npoints, :nelems, 0, 0] = 0
        self.sj[:npoints, :nelems, 0, 0] = 0
        self.tj[:npoints, :nelems, 0, 0] = 0
    else:
        self.rj[:npoints,:nelems,:,:] = 0
        self.sj[:npoints,:nelems,:,:] = 0
        self.tj[:npoints,:nelems,:,:] = 0

    return



def apply_operators_3d(dr, ds, dt, x):

    '''

    This function applies operators the same way as they are applied in NEK5000
    
    The only difference is that it is reversed, as this is python and we decided to leave that arrays as is

    this function is more readable in sem.py, where tensor optimization is not used.
    
    '''
    
    dshape = dr.shape
    xshape = x.shape
    xsize = xshape[2]*xshape[3]    

    # Reshape the operator in the r direction
    drt = dr.transpose(0,1,3,2) # This is just the transpose of the operator, leaving the first dimensions unaltered
    drt_s0 = drt.shape[2]
    drt_s1 = drt.shape[3]
    # Reshape the field to be consistent
    xreshape = x.reshape((xshape[0], xshape[1], int(xsize/drt_s0), drt_s0))
    # Apply the operator with einsum
    temp = np.einsum('ijkl,ijlm->ijkm', xreshape, drt)

    # Reshape the arrays as needed
    tempsize = temp.shape[2]*temp.shape[3]
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]

    # Apply in s direction
    temp = temp.reshape((xshape[0], xshape[1], ds_s1 , ds_s1, int(tempsize/(ds_s1**2))))
    temp = np.einsum('ijklm,ijkmn->ijkln' , ds.reshape((dshape[0], dshape[1], 1, ds_s0, ds_s1)),temp)
    
    # Reshape the arrays as needed
    tempsize = temp.shape[2]*temp.shape[3]*temp.shape[4]
    dt_s0 = dt.shape[2]
    dt_s1 = dt.shape[3]

    # Apply in t direction     

    temp = temp.reshape((xshape[0], xshape[1], dt_s1, int(tempsize/dt_s1)))
    temp = np.einsum('ijkl,ijlm->ijkm', dt, temp)

    # Reshape to proper size
    tempshape = temp.shape
    tempsize = temp.shape[2]*temp.shape[3]

    return temp.reshape((*tempshape[0:2], tempsize, 1))

def GLC_pwts(n):
    """ 
    Gauss-Lobatto-Chebyshev (GLC) points and weights over [-1,1]    
    Args: 
      `n`: int, number of nodes
    Returns 
       `x`: 1D numpy array of size `n`, nodes         
       `w`: 1D numpy array of size `n`, weights
    """
    def delt(i,n):
        del_=1.
        if i==0 or i==n-1:
           del_=0.5 
        return del_
    x=np.cos(np.arange(n)*pi/(n-1))
    w=np.zeros(n)    
    for i in range(n):
        tmp_=0.0
        for k in range(int((n-1)/2)):
            tmp_+=delt(2*k,n)/(1-4.*k**2)*np.cos(2*i*pi*k/(n-1))
        w[i]=tmp_*delt(i,n)*4/float(n-1)
    return x,w 

def GLL_pwts(n,eps=10**-8,maxIter=1000):
    """
    Generating `n `Gauss-Lobatto-Legendre (GLL) nodes and weights using the 
    Newton-Raphson iteration.
    Args:    
      `n`: int
         Number of GLL nodes
      `eps`: float (optional) 
         Min error to keep the iteration running
      `maxIter`: float (optional)
         Max number of iterations
    Outputs:
      `xi`: 1D numpy array of size `n`
         GLL nodes
      `w`: 1D numpy array of size `n`
         GLL weights
    Reference:
       Canuto C., Hussaini M. Y., Quarteroni A., Tang T. A., 
       "Spectral Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987.
       https://link.springer.com/book/10.1007/978-3-642-84108-8
    """
    V=np.zeros((n,n))  #Legendre Vandermonde Matrix
    #Initial guess for the nodes: GLC points
    xi,w_=GLC_pwts(n)
    iter_=0
    err=1000
    xi_old=xi
    while iter_<maxIter and err>eps:
        iter_+=1
        #Update the Legendre-Vandermonde matrix
        V[:,0]=1.
        V[:,1]=xi
        for j in range(2,n):
            V[:,j]=((2.*j-1)*xi*V[:,j-1] - (j-1)*V[:,j-2])/float(j)
        #Newton-Raphson iteration 
        xi=xi_old-(xi*V[:,n-1]-V[:,n-2])/(n*V[:,n-1])
        err=max(abs(xi-xi_old).flatten())
        xi_old=xi
    if (iter_>maxIter and err>eps):
       print('gllPts(): max iterations reached without convergence!')
    #Weights
    w=2./(n*(n-1)*V[:,n-1]**2.)
    return xi,w


def legendre_basis_at_xtest(n, xtest):

    '''
    The legendre basis depends on the element order and the points

    '''

    m = xtest.shape[0]
    m2 = xtest.shape[1]
 
    # Allocate space
    Leg=np.zeros((m, m2, n, 1))

        
    #First row is filled with 1 according to recursive formula
    Leg[:,:,0,0]=np.ones((m,m2,1,1))[:,:,0,0]
    #Second row is filled with x according to recursive formula
    Leg[:,:,1,0]=np.multiply(np.ones((m, m2, 1, 1)), xtest)[:,:,0,0]
    
    # Apply the recursive formula for all x_i
    # look for recursive formula here if you want to verify https://en.wikipedia.org/wiki/Legendre_polynomials
    for j in range(1,n-1):
        Leg[:,:, j+1,0]=((2*j+1)*xtest[:,:,0,0]*Leg[:,:,j,0]-j*Leg[:,:,j-1,0])/(j+1)

    return Leg


def legendre_basis_derivative_at_xtest(legtest, xtest):
    
    '''

    This is a slow implementaiton with a slow recursion. It does not need 
    special treatmet at the boundary

    '''

    ## Now find the derivative matrix D_N,ij=(dpi_j/dxi)_at_xi=xi_i
    ##https://en.wikipedia.org/wiki/Legendre_polynomials

    m = legtest.shape[0]
    m2 = xtest.shape[1]
    n = legtest.shape[2]
    
    # Allocate space
    D_N=np.zeros((m, m2, n, 1))

    # For first polynomial: derivatice is 0
    D_N[:,:,0,0]=np.zeros((m,m2,1,1))[:,:,0,0]
    # For second polynomial: derivatice is 1
    D_N[:,:,1,0]=np.ones((m, m2, 1, 1))[:,:,0,0]
    
    for j in range(1,n-1):
        for p in range(j, 0 - 1, -2):
            D_N[:, :,j+1, 0] += 2*legtest[:,:,p, 0]/(np.sqrt(2/(2*p+1))**2)

    return D_N

def get_reference_element(self):
    '''
    This routine creates a reference element containing all GLL points. 

    It also creates a place holder where data from future elements can be stored.

    Data is always of shape [points, elements, rows, columns]
    
    '''

    n = self.n
    max_pts = self.max_pts
    max_elems = self.max_elemes

    # Get the quadrature nodes
    x,w_=GLL_pwts(n) # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x_gll=np.copy(np.flip(x))     # Quadrature
    w=np.copy(np.flip(w_))    # Weights
 
    # Bounding boxes of the elements
    min_x = -1
    max_x = 1
    min_y = -1
    max_y = 1
    min_z = -1
    max_z = 1

    tmpx = np.zeros(len(x))
    tmpy = np.zeros(len(x))
    tmpz = np.zeros(len(x))
    for j in range(0,len(x)):
        tmpx[j]=((1-x_gll[j])/2*min_x+(1+x_gll[j])/2*max_x)
        tmpy[j]=((1-x_gll[j])/2*min_y+(1+x_gll[j])/2*max_y)
        tmpz[j]=((1-x_gll[j])/2*min_z+(1+x_gll[j])/2*max_z)

    x_3d=np.kron(np.ones((n)), np.kron(np.ones((n)), tmpx))
    y_3d=np.kron(np.ones((n)), np.kron(tmpy, np.ones((n))))
    z_3d=np.kron(tmpz, np.kron(np.ones((n)), np.ones((n))))        

    # Object atributes
    ## Allocate
    self.x_gll = np.zeros((n, 1, 1, 1))
    self.w_gll = np.zeros((n, 1, 1, 1))
    self.x_e = np.zeros((max_pts, max_elems, x_3d.size, 1))
    self.y_e = np.zeros((max_pts, max_elems, y_3d.size, 1))
    self.z_e = np.zeros((max_pts, max_elems, z_3d.size, 1))
    ## Assing values
    self.x_gll[:,0,0,0] = x_gll
    self.w_gll[:,0,0,0] = w
    self.x_e[:,:] = x_3d.reshape(-1,1)
    self.y_e[:,:] = y_3d.reshape(-1,1)
    self.z_e[:,:] = z_3d.reshape(-1,1)
    ## Copy data to the device
    self.x_gll_d = torch.as_tensor(self.x_gll, dtype=torch.float64, device=device)
    self.w_gll_d = torch.as_tensor(self.w_gll, dtype=torch.float64, device=device)
    self.x_e_d = torch.as_tensor(self.x_e, dtype=torch.float64, device=device)
    self.y_e_d = torch.as_tensor(self.y_e, dtype=torch.float64, device=device)
    self.z_e_d = torch.as_tensor(self.z_e, dtype=torch.float64, device=device) 
    
    return

## Create transformation matrices for the element (Only needs to be done once)
def get_basis_transformation_matrices(self):
    '''
    This routines generates the transformation matrices to be applied directly to the data.

    In principle this is not needed, but since many transformations need to be made, we store it.

    '''

    ## Legendre basis at the element gll points
    leg_gll = legendre_basis_at_xtest(self.n, self.x_gll) 
    leg_prm_gll = legendre_basis_derivative_at_xtest(leg_gll, self.x_gll)

    ## For the sake of simplicity, reshape the arrays only for this routine
    leg_gll = (leg_gll.transpose(3,1,2,0)).reshape((self.n, self.n))
    leg_prm_gll = (leg_prm_gll.transpose(3,1,2,0)).reshape((self.n, self.n))

    ## Transformation matrices for the element (1D)
    v_1d = leg_gll.T
    v_1d_inv = np.linalg.inv(v_1d)
    d_1d = leg_prm_gll.T
    ## Transformation matrices in 2d
    v_2d = np.kron(v_1d,v_1d)
    v_2d_inv = np.kron(v_1d_inv,v_1d_inv)
    ## Transformation matrices in 3d
    v_3d = np.kron(v_1d,v_2d)
    v_3d_inv = np.kron(v_1d_inv,v_2d_inv)

    self.v1d = v_1d.reshape((1, 1, self.n, self.n))
    self.v1d_inv = v_1d_inv.reshape((1, 1, self.n, self.n))

    # Assign attributes
    self.v = v_3d.reshape((1, 1, self.n**3, self.n**3))
    self.v_inv = v_3d_inv.reshape((1, 1, self.n**3, self.n**3))

    return 
