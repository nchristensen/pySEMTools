from abc import ABC, abstractmethod
import numpy as np
from math import pi
from scipy.special import legendre
from .single_point_helper_functions import *

class SinglePointInterpolator(ABC):
    def __init__(self, n):

        # Order of the element
        self.n = n

        # Attibutes concerning the reference element
        self.x_gll = None
        self.w_gll = None
        self.x_e = None
        self.y_e = None
        self.z_e = None
        get_reference_element(self)
 
        # Allocate relevant arrays
        self.xj = np.zeros(1)
        self.yj = np.zeros(1)
        self.zj = np.zeros(1)
        self.rj = np.zeros_like(self.xj)
        self.sj = np.zeros_like(self.yj)
        self.tj = np.zeros_like(self.zj)
        self.rstj = np.zeros((3,1))
        self.eps_rst = np.ones((3,1))
        self.jac = np.zeros((3,3))
        self.iterations = 0
        self.field_e = np.zeros_like(self.x_e)
        self.point_inside_element = False
         
        return
    
    def find_rst_from_xyz(self, xj, yj, zj, tol = np.finfo(np.double).eps*10, max_iterations = 1000):

        self.point_inside_element = False
        
        self.xj[0] = xj
        self.yj[0] = yj
        self.zj[0] = zj

        # Determine the initial conditions
        self.determine_initial_guess()

        # Use the newton method to identify the coordinates
        self.iterations = 0
        self.eps_rst[:,:] = 1
        while np.linalg.norm(self.eps_rst) > tol and self.iterations < max_iterations:

            # Update the guess
            self.rstj[0,0] = self.rj[0]
            self.rstj[1,0] = self.sj[0]
            self.rstj[2,0] = self.tj[0]

            # Estimate the xyz values from rst and also the jacobian (it is updated inside self.jac)
            xj_found, yj_found, zj_found = self.get_xyz_from_rst(self.rj[0], self.sj[0], self.tj[0])
            
            # Find the residuals and the jacobian inverse.
            self.eps_rst[0,0] = self.xj[0] - xj_found
            self.eps_rst[1,0] = self.yj[0] - yj_found
            self.eps_rst[2,0] = self.zj[0] - zj_found
            jac_inv = np.linalg.inv(self.jac)

            # Find the new guess
            self.rstj = self.rstj - (0 - jac_inv@self.eps_rst)

            # Update the values
            self.rj[0] = self.rstj[0,0]
            self.sj[0] = self.rstj[1,0]
            self.tj[0] = self.rstj[2,0]
            self.iterations += 1

        limit = 1 + np.finfo(np.single).eps
        if abs(self.rj[0]) <= limit and abs(self.sj[0]) <= limit and abs(self.tj[0]) <= limit:
            self.point_inside_element = True
        else:
            self.point_inside_element = False

        return self.rj[0], self.sj[0], self.tj[0]

    def interpolate_field_at_rst(self, rj, sj, tj, field_e, apply_1d_ops = True):

        r_j = np.ones((1))
        s_j = np.ones((1))
        t_j = np.ones((1))

        r_j[0] = rj
        s_j[0] = sj
        t_j[0] = tj
        
        self.field_e[:,0] = field_e[:,:,:].reshape(-1,1)[:,0]

        lk_r = lagInterp_matrix_at_xtest(self.x_gll,r_j)
        lk_s = lagInterp_matrix_at_xtest(self.x_gll,s_j)
        lk_t = lagInterp_matrix_at_xtest(self.x_gll,t_j)

        if not apply_1d_ops:
            lk_3d = np.kron(lk_t.T, np.kron(lk_s.T, lk_r.T))
            field_at_rst = (lk_3d@self.field_e)[0,0]
        elif apply_1d_ops:
            field_at_rst = apply_operators_3d(lk_r.T, lk_s.T, lk_t.T, self.field_e)[0,0]

        return field_at_rst

    def interpolate_field_at_rst_vector(self, rj, sj, tj, field_e, apply_1d_ops = True):

        r_j = np.ones((rj.size))
        s_j = np.ones((sj.size))
        t_j = np.ones((tj.size))

        r_j[:] = rj[:]
        s_j[:] = sj[:]
        t_j[:] = tj[:]
        
        self.field_e[:,0] = field_e[:,:,:].reshape(-1,1)[:,0]

        lk_r = lagInterp_matrix_at_xtest(self.x_gll,r_j)
        lk_s = lagInterp_matrix_at_xtest(self.x_gll,s_j)
        lk_t = lagInterp_matrix_at_xtest(self.x_gll,t_j)

        if not apply_1d_ops:
            lk_3d = np.kron(lk_t.T, np.kron(lk_s.T, lk_r.T))
            field_at_rst = (lk_3d@self.field_e)
        elif apply_1d_ops:
            field_at_rst = apply_operators_3d(lk_r.T, lk_s.T, lk_t.T, self.field_e)

        field_at_rst = field_at_rst.reshape((t_j.size, s_j.size, r_j.size))

        return field_at_rst
    
    @abstractmethod
    def project_element_into_basis(self):
        pass 
    
    @abstractmethod
    def get_xyz_from_rst(self):
        pass

    @abstractmethod
    def determine_initial_guess(self):
        pass


