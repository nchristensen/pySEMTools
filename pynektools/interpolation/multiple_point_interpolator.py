from abc import ABC, abstractmethod
import numpy as np
from .multiple_point_helper_functions_numpy import get_reference_element

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from math import pi
from scipy.special import legendre

class MultiplePointInterpolator(ABC):
    def __init__(self, n, max_pts = 1, max_elems = 1):
    
        # Order of the element
        self.n = n
        # Maximun of points to interpolate at the same time
        self.max_pts = max_pts
        # Maximun number of elements to interpolate at the same time
        self.max_elemes = max_elems

        # Attributes reference element
        self.x_gll = None
        self.w_gll = None
        self.x_e = None
        self.y_e = None
        self.z_e = None
        get_reference_element(self)
 
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
        self.field_e = np.zeros((max_pts, 1, self.x_e.shape[2], 1))
        self.point_inside_element = np.zeros((max_pts, max_elems, 1, 1), dtype=bool)
         
        return

    @abstractmethod
    def project_element_into_basis(self):
        pass

    @abstractmethod
    def get_xyz_from_rst(self):
        pass

    @abstractmethod
    def find_rst_from_xyz(self):
        pass

    @abstractmethod
    def interpolate_field_at_rst(self):
        pass

