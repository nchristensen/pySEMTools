""" Module that contains classes and methods that provide standard usefull quantities in SEM"""

from math import pi
import numpy as np
from pympler import asizeof


class Coef:
    """
    Class that contains arrays like mass matrix, jacobian, jacobian inverse, etc.

    This class can be used when mathematical operations such as derivation and integration is needed on the sem mesh.

    Parameters
    ----------
    msh : Mesh
        Mesh object.

    comm : Comm
        MPI comminicator object.
    
    get_area : bool, optional
        If True, the area integration weight and normal vectors will be calculated. (Default value = True).
    
    apply_1d_operators : bool, optional
        If True, the 1D operators will be applied instead of building 3D operators. (Default value = True).

    Attributes
    ----------
    drdx : ndarray
        component [0,0] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    drdy : ndarray
        component [0,1] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    drdz : ndarray
        component [0,2] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dsdx : ndarray
        component [1,0] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dsdy : ndarray
        component [1,1] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dsdz : ndarray
        component [1,2] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dtdx : ndarray
        component [2,0] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dtdy : ndarray
        component [2,1] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dtdz : ndarray
        component [2,2] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    B : ndarray
        Mass matrix for each point. shape is (nelv, lz, ly, lx).
    area : ndarray
        Area integration weight for each point in the facets. shape is (nelv, 6, ly, lx).
    nx : ndarray
        x component of the normal vector for each point in the facets. shape is (nelv, 6, ly, lx).
    ny : ndarray
        y component of the normal vector for each point in the facets. shape is (nelv, 6, ly, lx).
    nz : ndarray
        z component of the normal vector for each point in the facets. shape is (nelv, 6, ly, lx).

    Returns
    -------

    Examples
    --------
    Assuming you have a mesh object and MPI communicator object, you can initialize the Coef object as follows:

    >>> from pynektools import Coef
    >>> coef = Coef(msh, comm)
    """

    def __init__(self, msh, comm, get_area=True, apply_1d_operators = True):

        if comm.Get_rank() == 0:
            print("Initializing coef object")

        self.gdim = msh.gdim
        self.dtype = msh.x.dtype
        self.apply_1d_operators = apply_1d_operators
        
        self.v, self.vinv, self.w3, self.x, self.w = get_transform_matrix(
            msh.lx, msh.gdim, apply_1d_operators=apply_1d_operators, dtype=self.dtype
        )

        self.dr, self.ds, self.dt, self.dn = get_derivative_matrix(msh.lx, msh.gdim, dtype=self.dtype, apply_1d_operators=apply_1d_operators)

        # Find the components of the jacobian per point
        # jac(x,y,z) = [dxdr, dxds, dxdt ; dydr, dyds, dydt; dzdr, dzds, dzdt]
        self.dxdr = self.dudrst(msh.x, direction='r')
        self.dxds = self.dudrst(msh.x, direction='s')
        if msh.gdim > 2:
            self.dxdt = self.dudrst(msh.x, direction='t')

        self.dydr = self.dudrst(msh.y, direction='r')
        self.dyds = self.dudrst(msh.y, direction='s')
        if msh.gdim > 2:
            self.dydt = self.dudrst(msh.y, direction='t')

        if msh.gdim > 2:
            self.dzdr = self.dudrst(msh.z, direction='r')
            self.dzds = self.dudrst(msh.z, direction='s')
            self.dzdt = self.dudrst(msh.z, direction='t')

        self.drdx = np.zeros_like(self.dxdr, dtype=self.dtype)
        self.drdy = np.zeros_like(self.dxdr, dtype=self.dtype)
        if msh.gdim > 2:
            self.drdz = np.zeros_like(self.dxdr, dtype=self.dtype)

        self.dsdx = np.zeros_like(self.dxdr, dtype=self.dtype)
        self.dsdy = np.zeros_like(self.dxdr, dtype=self.dtype)
        if msh.gdim > 2:
            self.dsdz = np.zeros_like(self.dxdr, dtype=self.dtype)

        if msh.gdim > 2:
            self.dtdx = np.zeros_like(self.dxdr, dtype=self.dtype)
            self.dtdy = np.zeros_like(self.dxdr, dtype=self.dtype)
            self.dtdz = np.zeros_like(self.dxdr, dtype=self.dtype)

        # Find the jacobian determinant, its inverse inverse and mass matrix (3D)
        # This maps dxyz/drst
        # Gere we store the jacobian determinant as "jac"
        self.jac = np.zeros_like(self.dxdr, dtype=self.dtype)
        # This maps drst/dxyz
        self.jac_inv = np.zeros_like(self.dxdr, dtype=self.dtype)
        self.B = np.zeros_like(self.dxdr, dtype=self.dtype)

        if msh.gdim > 2:
            temp_mat = np.zeros((3, 3))
        else:
            temp_mat = np.zeros((2, 2))

        for e in range(0, msh.nelv):
            for k in range(0, msh.lz):
                for j in range(0, msh.ly):
                    for i in range(0, msh.lx):
                        temp_mat[0, 0] = self.dxdr[e, k, j, i]
                        temp_mat[0, 1] = self.dxds[e, k, j, i]
                        if msh.gdim > 2:
                            temp_mat[0, 2] = self.dxdt[e, k, j, i]

                        temp_mat[1, 0] = self.dydr[e, k, j, i]
                        temp_mat[1, 1] = self.dyds[e, k, j, i]
                        if msh.gdim > 2:
                            temp_mat[1, 2] = self.dydt[e, k, j, i]

                        if msh.gdim > 2:
                            temp_mat[2, 0] = self.dzdr[e, k, j, i]
                            temp_mat[2, 1] = self.dzds[e, k, j, i]
                            temp_mat[2, 2] = self.dzdt[e, k, j, i]

                        # Fill the jaconian determinant, its inverse and the mass matrix
                        self.jac[e, k, j, i] = np.linalg.det(temp_mat)
                        self.jac_inv[e, k, j, i] = 1 / self.jac[e, k, j, i]
                        self.B[e, k, j, i] = (
                            self.jac[e, k, j, i]
                            * (self.w3).reshape((msh.lz, msh.ly, msh.lx))[
                                k, j, i
                            ]
                        )
                        # Fill the terms for the jacobian inverse

                        # Find the components of the inverse of the jacobian per point
                        temp_mat_inv = np.linalg.inv(
                            temp_mat
                        )  # Note that here, 1/det(jac) is already performed

                        # Find the components of the jacobian per point
                        # jac_inv(r,s,t) = [drdx, drdy, drdz ; dsdx, dsdy, dsdz; dtdx, dtdy, dtdz]
                        self.drdx[e, k, j, i] = temp_mat_inv[0, 0]
                        self.drdy[e, k, j, i] = temp_mat_inv[0, 1]
                        if msh.gdim > 2:
                            self.drdz[e, k, j, i] = temp_mat_inv[0, 2]

                        self.dsdx[e, k, j, i] = temp_mat_inv[1, 0]
                        self.dsdy[e, k, j, i] = temp_mat_inv[1, 1]
                        if msh.gdim > 2:
                            self.dsdz[e, k, j, i] = temp_mat_inv[1, 2]

                        if msh.gdim > 2:
                            self.dtdx[e, k, j, i] = temp_mat_inv[2, 0]
                            self.dtdy[e, k, j, i] = temp_mat_inv[2, 1]
                            self.dtdz[e, k, j, i] = temp_mat_inv[2, 2]

        # Get area stuff only if mesh is 3D
        # Remember that the area described by two vectors is given by the norm of its cross product
        # i.e., norm(drxds)
        # Here we do that and then multiply by the weights.
        # Similar to what we do with the volume mass matrix.
        # Where we calculate the jacobian determinant
        # and then multiply with weights
        if msh.gdim > 2 and get_area:
            d1 = np.zeros((3), dtype=self.dtype)
            d2 = np.zeros((3), dtype=self.dtype)
            self.area = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=self.dtype)
            self.nx = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=self.dtype)
            self.ny = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=self.dtype)
            self.nz = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=self.dtype)

            # ds x dt
            for e in range(0, msh.nelv):
                for k in range(0, msh.lz):
                    for j in range(0, msh.ly):
                        weight = self.w[j] * self.w[k]

                        # For facet 1
                        d1[0] = self.dxds[e, k, j, 0]
                        d1[1] = self.dyds[e, k, j, 0]
                        d1[2] = self.dzds[e, k, j, 0]
                        d2[0] = self.dxdt[e, k, j, 0]
                        d2[1] = self.dydt[e, k, j, 0]
                        d2[2] = self.dzdt[e, k, j, 0]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 0, k, j] = norm * weight
                        self.nx[e, 0, k, j] = -cross[0] / norm
                        self.ny[e, 0, k, j] = -cross[1] / norm
                        self.nz[e, 0, k, j] = -cross[2] / norm

                        # For facet 2
                        d1[0] = self.dxds[e, k, j, msh.lx - 1]
                        d1[1] = self.dyds[e, k, j, msh.lx - 1]
                        d1[2] = self.dzds[e, k, j, msh.lx - 1]
                        d2[0] = self.dxdt[e, k, j, msh.lx - 1]
                        d2[1] = self.dydt[e, k, j, msh.lx - 1]
                        d2[2] = self.dzdt[e, k, j, msh.lx - 1]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 1, k, j] = norm * weight
                        self.nx[e, 1, k, j] = cross[0] / norm
                        self.ny[e, 1, k, j] = cross[1] / norm
                        self.nz[e, 1, k, j] = cross[2] / norm

            # dr x dt
            for e in range(0, msh.nelv):
                for k in range(0, msh.lz):
                    for i in range(0, msh.lx):
                        weight = self.w[i] * self.w[k]

                        # For facet 3
                        d1[0] = self.dxdr[e, k, 0, i]
                        d1[1] = self.dydr[e, k, 0, i]
                        d1[2] = self.dzdr[e, k, 0, i]
                        d2[0] = self.dxdt[e, k, 0, i]
                        d2[1] = self.dydt[e, k, 0, i]
                        d2[2] = self.dzdt[e, k, 0, i]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 2, k, i] = norm * weight
                        self.nx[e, 2, k, i] = cross[0] / norm
                        self.ny[e, 2, k, i] = cross[1] / norm
                        self.nz[e, 2, k, i] = cross[2] / norm

                        # For facet 4
                        d1[0] = self.dxdr[e, k, msh.ly - 1, i]
                        d1[1] = self.dydr[e, k, msh.ly - 1, i]
                        d1[2] = self.dzdr[e, k, msh.ly - 1, i]
                        d2[0] = self.dxdt[e, k, msh.ly - 1, i]
                        d2[1] = self.dydt[e, k, msh.ly - 1, i]
                        d2[2] = self.dzdt[e, k, msh.ly - 1, i]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 3, k, i] = norm * weight
                        self.nx[e, 3, k, i] = -cross[0] / norm
                        self.ny[e, 3, k, i] = -cross[1] / norm
                        self.nz[e, 3, k, i] = -cross[2] / norm

            # dr x ds
            for e in range(0, msh.nelv):
                for j in range(0, msh.ly):
                    for i in range(0, msh.lx):
                        weight = self.w[j] * self.w[i]

                        # For facet 5
                        d1[0] = self.dxdr[e, 0, j, i]
                        d1[1] = self.dydr[e, 0, j, i]
                        d1[2] = self.dzdr[e, 0, j, i]
                        d2[0] = self.dxds[e, 0, j, i]
                        d2[1] = self.dyds[e, 0, j, i]
                        d2[2] = self.dzds[e, 0, j, i]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 4, j, i] = norm * weight
                        self.nx[e, 4, j, i] = -cross[0] / norm
                        self.ny[e, 4, j, i] = -cross[1] / norm
                        self.nz[e, 4, j, i] = -cross[2] / norm

                        # For facet 6
                        d1[0] = self.dxdr[e, msh.lz - 1, j, i]
                        d1[1] = self.dydr[e, msh.lz - 1, j, i]
                        d1[2] = self.dzdr[e, msh.lz - 1, j, i]
                        d2[0] = self.dxds[e, msh.lz - 1, j, i]
                        d2[1] = self.dyds[e, msh.lz - 1, j, i]
                        d2[2] = self.dzds[e, msh.lz - 1, j, i]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 5, j, i] = norm * weight
                        self.nx[e, 5, j, i] = cross[0] / norm
                        self.ny[e, 5, j, i] = cross[1] / norm
                        self.nz[e, 5, j, i] = cross[2] / norm

        if comm.Get_rank() == 0:
            print(f"coef data is of type: {self.B.dtype}")

    def __memory_usage__(self, comm):
        """
        Print the memory usage of the object.

        This function is used to print the memory usage of the object.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.

        Returns
        -------
        None

        """
        memory_usage = asizeof.asizeof(self) / (1024**2)  # Convert bytes to MB
        print(f"Rank: {comm.Get_rank()} - Memory usage of Coef: {memory_usage} MB")

    def __memory_usage_per_attribute__(self, comm, print_data=True):
        """
        Store and print the memory usage of each attribute of the object.

        This function is used to print the memory usage of each attribute of the object.
        The results are stored in the mem_per_attribute attribute.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.
        print_data : bool, optional
            If True, the memory usage of each attribute will be printed.

        Returns
        -------
        None

        """
        attributes = dir(self)
        non_callable_attributes = [
            attr
            for attr in attributes
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        size_per_attribute = [
            asizeof.asizeof(getattr(self, attr)) / (1024**2)
            for attr in non_callable_attributes
        ]  # Convert bytes to MB

        self.mem_per_attribute = dict()
        for i, attr in enumerate(non_callable_attributes):
            self.mem_per_attribute[attr] = size_per_attribute[i]

            if print_data:
                print(
                    f"Rank: {comm.Get_rank()} - Memory usage of coef attr - {attr}: {size_per_attribute[i]} MB"
                )

    def dudrst(self, field, direction='r'):
        '''
        Perform derivative with respect to reference coordinate r/s/t.

        Used to perform the derivative in the reference coordinates

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).
        
        direction : str
            Direction to take the derivative. Can be 'r', 's', or 't'. (Default value = 'r').

        Returns
        -------
        ndarray
            Derivative of the field with respect to r/s/t. Shape is the same as the input field.
        '''
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]
        
        if not self.apply_1d_operators:
            if direction == 'r':
                return self.dudrst_3d_operator(field, self.dr)
            elif direction == 's':
                return self.dudrst_3d_operator(field, self.ds)
            elif direction == 't':
                return self.dudrst_3d_operator(field, self.dt)
            else:
                raise ValueError("Invalid direction. Should be 'r', 's', or 't'")
        elif self.apply_1d_operators:
            if direction == 'r':
                if self.gdim == 2:
                    return self.dudrst_1d_operator(field, self.dr, np.eye(ly))
                elif self.gdim == 3:
                    return self.dudrst_1d_operator(field, self.dr, np.eye(ly), np.eye(lz))
            elif direction == 's':
                if self.gdim == 2:
                    return self.dudrst_1d_operator(field, np.eye(lx), self.ds)
                elif self.gdim == 3:
                    return self.dudrst_1d_operator(field, np.eye(lx), self.ds, np.eye(lz))
            elif direction == 't':
                return self.dudrst_1d_operator(field, np.eye(lx), np.eye(ly), self.dt)
            else:
                raise ValueError("Invalid direction. Should be 'r', 's', or 't'")

    def dudrst_3d_operator(self, field, dr):
        """
        Perform derivative with respect to reference coordinate r.

        This method uses derivation matrices from the lagrange polynomials at the GLL points.

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).
        dr : ndarray
            Derivative matrix in the r/s/t direction to apply to each element. Shape should be (lx*ly*lz, lx*ly*lz).

        Returns
        -------
        ndarray
            Derivative of the field with respect to r/s/t. Shape is the same as the input field.

        Examples
        --------
        Assuming you have a Coef object

        >>> dxdr = coef.dudrst(x, coef.dr)
        """
        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        dudrst = np.zeros_like(field, dtype=field.dtype)

        for e in range(0, nelv):
            tmp = field[e, :, :, :].reshape(-1, 1)
            dtmp = dr @ tmp
            dudrst[e, :, :, :] = dtmp.reshape((lz, ly, lx))

        return dudrst

    def dudrst_1d_operator(self, field, dr, ds, dt=None):
        """
        Perform derivative with respect to reference coordinate r.

        This method uses derivation matrices from the lagrange polynomials at the GLL points.

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).
        dr : ndarray
            Derivative matrix in the r direction to apply to each element. Shape should be (lx, lx).
        
        ds : ndarray
            Derivative matrix in the s direction to apply to each element. Shape should be (ly, ly).
        
        dt : ndarray
            Derivative matrix in the t direction to apply to each element. Shape should be (lz, lz).
            This is optional. If none is passed, it is assumed that the field is 2D.

        Returns
        -------
        ndarray
            Derivative of the field with respect to r/s/t. Shape is the same as the input field.

        Examples
        --------
        Assuming you have a Coef object

        >>> dxdr = coef.dudrst(x, coef.dr)
        """
        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        dudrst = np.zeros_like(field, dtype=field.dtype)

        for e in range(0, nelv):
            tmp = field[e, :, :, :].reshape(-1, 1)
            dtmp = apply_1d_operators(tmp, dr, ds, dt)
            dudrst[e, :, :, :] = dtmp.reshape((lz, ly, lx))

        return dudrst

    def dudxyz(self, field, drdx, dsdx, dtdx=None):
        """
        Perform derivative with respect to physical coordinate x,y,z.

        This method uses the chain rule, first evaluating derivatives with respect to
        rst, then multiplying by the inverse of the jacobian to map to xyz.

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).
        drdx : ndarray
            Derivative of the reference coordinates with respect to x, i.e.,
            first entry in the appropiate row of the jacobian inverse.
            Shape should be the same as the field.
        dsdx : ndarray
            Derivative of the reference coordinates with respect to y, i.e.,
            second entry in the appropiate row of the jacobian inverse.
            Shape should be the same as the field.
        dtdx : ndarray
            Derivative of the reference coordinates with respect to z, i.e.,
            third entry in the appropiate row of the jacobian inverse.
            Shape should be the same as the field.
            (Default value = None)
            Only valid for 3D fields.

        Returns
        -------
        ndarray
            Derivative of the field with respect to x,y,z. Shape is the same as the input field.

        Examples
        --------
        Assuming you have a Coef object and are working on a 3d field:

        >>> dudx = coef.dudxyz(u, coef.drdx, coef.dsdx, coef.dtdx)
        """

        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]
        dudxyz = np.zeros_like(field, dtype=field.dtype)

        dfdr = self.dudrst(field, direction='r')
        dfds = self.dudrst(field, direction='s')
        if self.gdim > 2:
            dfdt = self.dudrst(field, direction='t')

        # NOTE: DO NOT NEED TO MULTIPLY BY INVERSE OF JACOBIAN DETERMINAT.
        # THIS STEP IS ALREADY DONE IF YOU CALCULATED THE INVERSE WITH NUMPY
        # Here we multiply the derivative in reference element with the respective
        # row of the jacobian
        if self.gdim > 2:
            for e in range(0, nelv):
                for k in range(0, lz):
                    for j in range(0, ly):
                        for i in range(0, lx):
                            # dudxyz[e,k,j,i] = self.jac_inv[e, k, j, i] *
                            # (dfdr[e, k, j, i] *drdx[e, k, j, i]  + dfds[e, k, j, i] *
                            # dsdx[e, k, j, i]  + dfdt[e, k, j, i] *dtdx[e, k, j, i] )
                            dudxyz[e, k, j, i] = (
                                dfdr[e, k, j, i] * drdx[e, k, j, i]
                                + dfds[e, k, j, i] * dsdx[e, k, j, i]
                                + dfdt[e, k, j, i] * dtdx[e, k, j, i]
                            )
        else:
            for e in range(0, nelv):
                for k in range(0, lz):
                    for j in range(0, ly):
                        for i in range(0, lx):
                            # dudxyz[e,k,j,i] = self.jac_inv[e, k, j, i] *
                            # ( dfdr[e, k, j, i] * drdx[e, k, j, i] +
                            # dfds[e, k, j, i] * dsdx[e, k, j, i] )
                            dudxyz[e, k, j, i] = (
                                dfdr[e, k, j, i] * drdx[e, k, j, i]
                                + dfds[e, k, j, i] * dsdx[e, k, j, i]
                            )

        return dudxyz

    def glsum(self, a, comm, datype=np.double):
        """
        Peform global summatin of given qunaitity a using MPI.

        This method uses MPI to sum over all MPI ranks. It works with any numpy array shape and returns one value.

        Parameters
        ----------
        a : ndarray
            Quantity to sum over all mpiranks.
        comm : Comm
            MPI communicator object.
        datype : numpy.dtype
             (Default value = np.double).

        Returns
        -------
        float
            Sum of the quantity a over all MPI ranks.

        Examples
        --------
        Assuming you have a Coef object and are working on a 3d field:

        >>> volume = coef.glsum(coef.B, comm)
        """

        sendbuf = np.ones((1), datype)
        sendbuf[0] = np.sum(a)
        recvbuf = np.zeros((1), datype)
        comm.Allreduce(sendbuf, recvbuf)

        return recvbuf[0]

    def dssum(self, field, msh):
        """
        Peform average of given field over shared points in each rank.

        This method averages the field over shared points in the same rank. It uses the connectivity data in the mesh object.
        dssum might be a missleading name.

        Parameters
        ----------
        field : ndarray
            Field to average over shared points.
        msh : Mesh
            Pynektools Mesh object.

        Returns
        -------
        ndarray
            Input field with shared points averaged with shared points in the SAME rank.

        Examples
        --------
        Assuming you have a Coef object and are working on a 3d field:

        >>> dudx = coef.dssum(dudx, msh)
        """

        if msh.create_connectivity:
            if msh.lz > 1:
                z_ind = [0, msh.lz - 1]
            else:
                z_ind = [0]
            tmp = np.copy(field)

            for e in range(0, msh.nelv):
                # loop through all faces (3 loops required)
                for k in z_ind:
                    for j in range(0, msh.ly):
                        for i in range(0, msh.lx):
                            point = (
                                msh.x[e, k, j, i],
                                msh.y[e, k, j, i],
                                msh.z[e, k, j, i],
                            )
                            point = hash(point)
                            shared_points = msh.coord_hash_to_shared_map[point]
                            shared_points = nonlinear_index(
                                shared_points, msh.lx, msh.ly, msh.lz
                            )
                            field_at_shared = np.array(
                                [
                                    tmp[shared_points[l]]
                                    for l in range(len(shared_points))
                                ]
                            )
                            field[e, k, j, i] = np.mean(field_at_shared)

                for j in [0, msh.ly - 1]:
                    for k in range(msh.lz):
                        for i in range(msh.lx):
                            point = (
                                msh.x[e, k, j, i],
                                msh.y[e, k, j, i],
                                msh.z[e, k, j, i],
                            )
                            point = hash(point)
                            shared_points = msh.coord_hash_to_shared_map[point]
                            shared_points = nonlinear_index(
                                shared_points, msh.lx, msh.ly, msh.lz
                            )
                            field_at_shared = np.array(
                                [
                                    tmp[shared_points[l]]
                                    for l in range(len(shared_points))
                                ]
                            )
                            field[e, k, j, i] = np.mean(field_at_shared)

                for i in [0, msh.lx - 1]:
                    for k in range(msh.lz):
                        for j in range(msh.ly):
                            point = (
                                msh.x[e, k, j, i],
                                msh.y[e, k, j, i],
                                msh.z[e, k, j, i],
                            )
                            point = hash(point)
                            shared_points = msh.coord_hash_to_shared_map[point]
                            shared_points = nonlinear_index(
                                shared_points, msh.lx, msh.ly, msh.lz
                            )
                            field_at_shared = np.array(
                                [
                                    tmp[shared_points[l]]
                                    for l in range(len(shared_points))
                                ]
                            )
                            field[e, k, j, i] = np.mean(field_at_shared)

        else:
            print("Mesh does not have connectivity data. Returning unmodified array")

        return field


# -----------------------------------------------------------------------


## Define functions for the calculation of the quadrature points (Taken from the lecture notes)
def GLC_pwts(n, dtype=np.double):
    """Gauss-Lobatto-Chebyshev (GLC) points and weights over [-1,1]

    Parameters
    ----------
    `n` :
        int, number of nodes
        Returns
        `x`: 1D numpy array of size `n`, nodes
        `w`: 1D numpy array of size `n`, weights
    n :


    Returns
    -------


    """

    def delt(i, n):
        """Helper function

        Parameters
        ----------
        i : int

        n : int


        Returns
        -------


        """
        del_ = 1.0
        if i == 0 or i == n - 1:
            del_ = 0.5
        return del_

    x = np.cos(np.arange(n, dtype=dtype) * pi / (n - 1))
    w = np.zeros(n, dtype=dtype)
    for i in range(n):
        tmp_ = 0.0
        for k in range(int((n - 1) / 2)):
            tmp_ += delt(2 * k, n) / (1 - 4.0 * k**2) * np.cos(2 * i * pi * k / (n - 1))
        w[i] = tmp_ * delt(i, n) * 4 / float(n - 1)
    return x, w


def GLL_pwts(n, eps=10**-8, max_iter=1000, dtype=np.double):
    """Generating `n` Gauss-Lobatto-Legendre (GLL) nodes and weights using the
    Newton-Raphson iteration.

    Parameters
    ----------
    `n` :
        int
        Number of GLL nodes
    `eps` :
        float (optional)
        Min error to keep the iteration running
    `maxIter` :
        float (optional)
        Max number of iterations
        Outputs:
    `xi` :
        1D numpy array of size `n`
        GLL nodes
    `w` :
        1D numpy array of size `n`
        GLL weights
        Reference:
        Canuto C., Hussaini M. Y., Quarteroni A., Tang T. A.,
        "Spectral Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987.
        https://link.springer.com/book/10.1007/978-3-642-84108-8
    n :

    eps :
        (Default value = 10**-8)
    max_iter :
        (Default value = 1000)

    Returns
    -------


    """
    V = np.zeros((n, n), dtype=dtype)  # Legendre Vandermonde Matrix
    # Initial guess for the nodes: GLC points
    xi, _ = GLC_pwts(n, dtype=dtype)
    iter_ = 0
    err = 1000
    xi_old = xi
    while iter_ < max_iter and err > eps:
        iter_ += 1
        # Update the Legendre-Vandermonde matrix
        V[:, 0] = 1.0
        V[:, 1] = xi
        for j in range(2, n):
            V[:, j] = (
                (2.0 * j - 1) * xi * V[:, j - 1] - (j - 1) * V[:, j - 2]
            ) / float(j)
        # Newton-Raphson iteration
        xi = xi_old - (xi * V[:, n - 1] - V[:, n - 2]) / (n * V[:, n - 1])
        err = max(abs(xi - xi_old).flatten())
        xi_old = xi
    if iter_ > max_iter and err > eps:
        print("gllPts(): max iterations reached without convergence!")
    # Weights
    w = 2.0 / (n * (n - 1) * V[:, n - 1] ** 2.0)
    return xi, w


def get_transform_matrix(n, dim, apply_1d_operators=False, dtype=np.double):
    """
    get transformation matrix to Legendre space of given order and dimension

    Parameters
    ----------
    n : int
        Polynomial degree (order - 1).

    dim : int
        Dimension of the problem.

    Returns
    -------
    vv : ndarray
        Transformation matrix to Legendre space.
    vvinv : ndarray
        Inverse of the transformation matrix.
    w3 : ndarray
        3D weights.
    x : ndarray
        Quadrature nodes.
    w : ndarray
        Quadrature weights.
    """
    # Get the quadrature nodes
    x, w_ = GLL_pwts(
        n, dtype=dtype
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x = np.flip(x)
    w = np.flip(w_)

    # Create a diagonal matrix
    ww = np.eye((n), dtype=dtype)
    for i in range(0, n):
        ww[i, i] = w[i]

    ## First we need the legendre polynomials
    # order of the polynomials
    p = n
    # Create a counter for the loops
    p_v = np.arange(p)

    # get the legendre polynomial matrix

    # The polynomials are stored in a matrix with the following structure:
    #  |  pi_0(x0)  ... pi_0(x_n)
    #  |  pi_1(x0)  ... pi_1(x_n)
    #  |  ...
    #  |  pi_p(x0)  ... pi_p(x_n)
    #  The acending rows represent accending polynomial order,
    #  the different columns represent different x_i

    # Allocate space
    leg = np.zeros((p, p), dtype=dtype)
    # First row is filled with 1 according to recursive formula
    leg[0, :] = np.ones((1, p))
    # Second row is filled with x according to recursive formula
    leg[1, :] = np.multiply(np.ones((1, p)), x)

    # Apply the recursive formula for all x_i
    for j in range(1, len(p_v) - 1):
        for k_ in p_v:
            leg[j + 1, k_] = ((2 * j + 1) * x[k_] * leg[j, k_] - j * leg[j - 1, k_]) / (
                j + 1
            )

    leg = leg.T  # nek and I transpose it for transform

    # Scaling factor as in books
    delta = np.ones(n, dtype=dtype)
    for i in range(0, n):
        # delta[i]=2/(2*i+1)       #it is the same both ways
        delta[i] = 2 / (2 * (i + 1) - 1)
    delta[n - 1] = 2 / (n - 1)
    # print(delta)
    # Scaling factor to normalize
    for i in range(0, n):
        delta[i] = np.sqrt(1 / delta[i])

    # apply the scaling factor
    for i in range(0, n):
        for j in range(0, n):
            leg[i, j] = leg[i, j] * delta[j]

    # AA = np.matmul(leg.T, np.matmul(ww, leg))

    # 2d transformation matrix
    v = leg
    vinv = leg.T @ ww
    if not apply_1d_operators: 
        v2d = np.kron(v, v)
        vinv2d = np.kron(vinv, vinv)
    else:
        v2d = v
        vinv2d = vinv

    # 3d transformation matrix
    v = leg
    vinv = leg.T @ ww
    if not apply_1d_operators: 
        v3d = np.kron(v, np.kron(v, v))
        vinv3d = np.kron(vinv, np.kron(vinv, vinv))
    else:
        v3d = v
        vinv3d = vinv

    if dim == 1:
        vv = v.astype(dtype)
        vvinv = vinv.astype(dtype)
        w3 = w
    elif dim == 2:
        vv = v2d.astype(dtype)
        vvinv = vinv2d.astype(dtype)
        w3 = np.diag(np.kron(ww, ww)).copy()
    else:
        vv = v3d.astype(dtype)
        vvinv = vinv3d.astype(dtype)
        w3 = np.diag(np.kron(ww, np.kron(ww, ww))).copy()

    return vv, vvinv, w3, x, w


def get_derivative_matrix(n, dim, dtype=np.double, apply_1d_operators=False):
    """
    Derivative matrix of Lagrange polynomials a GLL points.

    Parameters
    ----------
    n : int
        Polynomial degree (order - 1).

    dim : int
        Dimension of the problem.
    
    apply_1d_operators : bool, optional
        If True, the 1D operators will be applied instead of constructing 3d.

    dtype : numpy.dtype, optional
        Data type of the output matrices.

    Returns
    -------
    dx : ndarray
        Derivation matrix wrt r direction.
    dy : ndarray
        Derivation matrix wrt s direction.
    dz : ndarray
        Derivation matrix wrt t direction.
    d_n : ndarray
        Derivation matrix in 1D.
    """
    # Get the quadrature nodes
    x, w_ = GLL_pwts(
        n, dtype=dtype
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x = np.flip(x)
    w = np.flip(w_)

    # Create a diagonal matrix
    ww = np.eye(n, dtype=dtype)
    for i in range(0, n):
        ww[i, i] = w[i]

    ## First we need the legendre polynomials
    # order of the polynomials
    p = n
    # Create a counter for the loops
    p_v = np.arange(p)

    # get the legendre polynomial matrix

    # The polynomials are stored in a matrix with the following structure:
    #  |  pi_0(x0)  ... pi_0(x_n)
    #  |  pi_1(x0)  ... pi_1(x_n)
    #  |  ...
    #  |  pi_p(x0)  ... pi_p(x_n)
    #  The acending rows represent accending polynomial order,
    #  the different columns represent different x_i

    # Allocate space
    leg = np.zeros((p, p), dtype=dtype)
    # First row is filled with 1 according to recursive formula
    leg[0, :] = np.ones((1, p))
    # Second row is filled with x according to recursive formula
    leg[1, :] = np.multiply(np.ones((1, p)), x)

    # Apply the recursive formula for all x_i
    for j in range(1, len(p_v) - 1):
        for k_ in p_v:
            leg[j + 1, k_] = ((2 * j + 1) * x[k_] * leg[j, k_] - j * leg[j - 1, k_]) / (
                j + 1
            )

    d_n = np.zeros((p, p), dtype=dtype)

    # Simply apply the values as given in the book
    for i in range(0, len(p_v)):
        for j in range(0, len(p_v)):
            if i != j:
                d_n[i, j] = (leg[p - 1, i] / leg[p - 1, j]) * (1 / (x[i] - x[j]))
            if i == 0 and j == 0:
                d_n[i, j] = -(((p - 1) + 1) * (p - 1)) / 4
            if i == (p - 1) and j == (p - 1):
                d_n[i, j] = (((p - 1) + 1) * (p - 1)) / 4
            if i == j and i != 0 and i != (p - 1):
                d_n[i, j] = 0

    if dim == 1:
        dx = d_n
        dy = None
        dz = None
    elif dim == 2:
        if not apply_1d_operators:
            dx2d = np.kron(np.eye(p), d_n)
            dy2d = np.kron(d_n, np.eye(p))
        else:
            dx2d = d_n
            dy2d = d_n

        dx = dx2d
        dy = dy2d
        dz = None
    else:
        if not apply_1d_operators:
            dx3d = np.kron(np.eye(p), np.kron(np.eye(p), d_n))
            dy3d = np.kron(np.eye(p), np.kron(d_n, np.eye(p)))
            dz3d = np.kron(d_n, np.kron(np.eye(p), np.eye(p)))
        else:
            dx3d = d_n
            dy3d = d_n
            dz3d = d_n

        dx = dx3d.astype(dtype)
        dy = dy3d.astype(dtype)
        dz = dz3d.astype(dtype)

    return dx, dy, dz, d_n


def nonlinear_index(linear_index_, lx, ly, lz):
    """
    Map 1d index to 4d

    This is an inverse of linear index.

    Parameters
    ----------
    linear_index_ : list
        List of 1d linear indices.
    lx : int
        Polynomial degree in x direction.
    ly : int
        Polynomial degree in y direction.
    lz : int
        Polynomial degree in z direction.
    Returns
    -------
    list
        List of 4d non linear indices correspoinf to the linear indices.
    """
    indices = []
    for list_ in linear_index_:
        index = np.zeros(4, dtype=int)
        lin_idx = list_
        index[3] = lin_idx / (lx * ly * lz)
        index[2] = (lin_idx - (lx * ly * lz) * index[3]) / (lx * ly)
        index[1] = (lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2]) / lx
        index[0] = (
            lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2] - lx * index[1]
        )
        ind = (index[3], index[2], index[1], index[0])
        indices.append(ind)

    return indices


def apply_1d_operators(x, dr, ds, dt=None):

    if not isinstance(dt, type(None)):
        return apply_operators_3d(dr, ds, dt, x)
    else:
        return apply_operators_2d(dr, ds, x)

def apply_operators_2d(dr, ds, x):
    """This function applies operators the same way as they are applied in NEK5000
    The only difference is that it is reversed, as this is
    python and we decided to leave that arrays as is"""

    # Apply in r direction
    temp = x.reshape((int(x.size / dr.T.shape[0]), dr.T.shape[0])) @ dr.T

    # Apply in s direction
    temp = ds @ temp.reshape(ds.shape[1], (int(temp.size / ds.shape[1])))

    return temp.reshape(-1, 1)

def apply_operators_3d(dr, ds, dt, x):
    """This function applies operators the same way as they are applied in NEK5000
    The only difference is that it is reversed, as this is
    python and we decided to leave that arrays as is"""

    # Apply in r direction
    temp = x.reshape((int(x.size / dr.T.shape[0]), dr.T.shape[0])) @ dr.T

    # Apply in s direction
    temp = temp.reshape((ds.shape[1], ds.shape[1], int(temp.size / (ds.shape[1] ** 2))))
    ### The nek5000 way uses a for loop
    ## temp2 = np.zeros((ds.shape[1], ds.shape[0],
    # int(temp.size/(ds.shape[1]**2))))
    # This is needed because dimensions could reduce
    ## for k in range(0, temp.shape[0]):
    ##     temp2[k,:,:] = ds@temp[k,:,:]
    ### We can do it optimized in numpy if we reshape the operator. This way it can broadcast
    temp = ds.reshape((1, ds.shape[0], ds.shape[1])) @ temp

    # Apply in t direction
    temp = dt @ temp.reshape(dt.shape[1], (int(temp.size / dt.shape[1])))

    return temp.reshape(-1, 1)