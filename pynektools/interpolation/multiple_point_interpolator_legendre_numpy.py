import numpy as np

from .multiple_point_interpolator import MultiplePointInterpolator
from .multiple_point_helper_functions_numpy import get_basis_transformation_matrices, apply_operators_3d, legendre_basis_at_xtest, legendre_basis_derivative_at_xtest, lagInterp_matrix_at_xtest

class LegendreInterpolator(MultiplePointInterpolator):
    def __init__(self, n, max_pts = 1, max_elems = 1):
        # Initialize parent class
        super().__init__(n, max_pts = max_pts, max_elems=max_elems)
    
        # Get reference element
        self.v1d      = None
        self.v1d_inv  = None
        self.v     = None
        self.v_inv = None 
        get_basis_transformation_matrices(self)
                 
        return

    def project_element_into_basis(self, x_e, y_e, z_e, apply_1d_ops = True):
        
        npoints = x_e.shape[0]
        nelems = x_e.shape[1]
        n = x_e.shape[2]*x_e.shape[3]*x_e.shape[4]

        # Assing the inputs to proper formats
        self.x_e[:npoints,:nelems,:,:] = x_e.reshape(npoints, nelems, n, 1)[:,:,:,:]
        self.y_e[:npoints,:nelems,:,:] = y_e.reshape(npoints, nelems, n, 1)[:,:,:,:]
        self.z_e[:npoints,:nelems,:,:] = z_e.reshape(npoints, nelems, n, 1)[:,:,:,:]

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
        ortho_basis_rj = legendre_basis_at_xtest(n, self.rj[:npoints,:nelems,:,:])
        ortho_basis_sj = legendre_basis_at_xtest(n, self.sj[:npoints,:nelems,:,:])
        ortho_basis_tj = legendre_basis_at_xtest(n, self.tj[:npoints,:nelems,:,:])

        ortho_basis_prm_rj = legendre_basis_derivative_at_xtest(ortho_basis_rj, self.rj[:npoints,:nelems,:,:])
        ortho_basis_prm_sj = legendre_basis_derivative_at_xtest(ortho_basis_sj, self.sj[:npoints,:nelems,:,:])
        ortho_basis_prm_tj = legendre_basis_derivative_at_xtest(ortho_basis_tj, self.tj[:npoints,:nelems,:,:])
                
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

        '''

        Find rst coordinates from a given xyz group of points.
        Note that this function needs to be called after the element has been projected into the basis.

        '''

        self.point_inside_element[:,:,:,:] = False
        
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

        # Check if points are inside the element
        limit = 1 + np.finfo(np.single).eps
        t1 = (abs(self.rj[:npoints, :nelems, 0, 0]) <= limit).reshape(npoints, nelems, 1, 1)
        t2 = (abs(self.sj[:npoints, :nelems, 0, 0]) <= limit).reshape(npoints, nelems, 1, 1)
        t3 = (abs(self.tj[:npoints, :nelems, 0, 0]) <= limit).reshape(npoints, nelems, 1, 1)

        # Pointwise comparison
        self.point_inside_element[:npoints, :nelems, :, :] = (t1 & t2 & t3)

        return self.rj[:npoints, :nelems], self.sj[:npoints, :nelems], self.tj[:npoints, :nelems]

    def interpolate_field_at_rst(self, rj, sj, tj, field_e, apply_1d_ops = True):

        '''
        Interpolate each point in a given field. EACH POINT RECIEVES ONE FIELD! SO FIELDS MIGHT BE DUPLICATED

        '''
        npoints = rj.shape[0]
        nelems  = rj.shape[1]
        n = field_e.shape[2]*field_e.shape[3]*field_e.shape[4]

        self.rj[:npoints,:nelems,:,:] = rj[:,:,:,:]
        self.sj[:npoints,:nelems,:,:] = sj[:,:,:,:]
        self.tj[:npoints,:nelems,:,:] = tj[:,:,:,:]

        # Assing the inputs to proper formats
        self.field_e[:npoints,:nelems,:,:] = field_e.reshape(npoints, nelems, n, 1)[:,:,:,:]

        lk_r = lagInterp_matrix_at_xtest(self.x_gll, self.rj[:npoints,:nelems,:,:])
        lk_s = lagInterp_matrix_at_xtest(self.x_gll, self.sj[:npoints,:nelems,:,:])
        lk_t = lagInterp_matrix_at_xtest(self.x_gll, self.tj[:npoints,:nelems,:,:])

        if not apply_1d_ops:
            raise RuntimeError("Only worrking by applying 1d operators")
        elif apply_1d_ops:
            field_at_rst = apply_operators_3d(lk_r.transpose(0,1,3,2), lk_s.transpose(0,1,3,2), lk_t.transpose(0,1,3,2), self.field_e[:npoints, :nelems, :, :])   

        return field_at_rst


def determine_initial_guess(self, npoints = 1, nelems = 1):
    '''
    Note: Find a way to evaluate if this routine does help. 
    It might be that this is not such a good way of making the guess.
    '''

    self.rj[:npoints,:nelems,:,:] = 0
    self.sj[:npoints,:nelems,:,:] = 0
    self.tj[:npoints,:nelems,:,:] = 0

    return

