""" Module that contains classes and methods that provide standard usefull quantities in SEM"""

from math import pi
import numpy as np


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

    def __init__(self, msh, comm):
        self.gdim = msh.gdim

        self.v, self.vinv, self.w3, self.x, self.w = get_transform_matrix(
            msh.lx, msh.gdim
        )

        self.dr, self.ds, self.dt, self.dn = get_derivative_matrix(msh.lx, msh.gdim)

        # Find the components of the jacobian per point
        # jac(x,y,z) = [dxdr, dxds, dxdt ; dydr, dyds, dydt; dzdr, dzds, dzdt]
        self.dxdr = self.dudrst(msh.x, self.dr)
        self.dxds = self.dudrst(msh.x, self.ds)
        if msh.gdim > 2:
            self.dxdt = self.dudrst(msh.x, self.dt)

        self.dydr = self.dudrst(msh.y, self.dr)
        self.dyds = self.dudrst(msh.y, self.ds)
        if msh.gdim > 2:
            self.dydt = self.dudrst(msh.y, self.dt)

        if msh.gdim > 2:
            self.dzdr = self.dudrst(msh.z, self.dr)
            self.dzds = self.dudrst(msh.z, self.ds)
            self.dzdt = self.dudrst(msh.z, self.dt)

        self.drdx = np.zeros_like(self.dxdr, dtype=np.double)
        self.drdy = np.zeros_like(self.dxdr, dtype=np.double)
        if msh.gdim > 2:
            self.drdz = np.zeros_like(self.dxdr, dtype=np.double)

        self.dsdx = np.zeros_like(self.dxdr, dtype=np.double)
        self.dsdy = np.zeros_like(self.dxdr, dtype=np.double)
        if msh.gdim > 2:
            self.dsdz = np.zeros_like(self.dxdr, dtype=np.double)

        if msh.gdim > 2:
            self.dtdx = np.zeros_like(self.dxdr, dtype=np.double)
            self.dtdy = np.zeros_like(self.dxdr, dtype=np.double)
            self.dtdz = np.zeros_like(self.dxdr, dtype=np.double)

        # Find the jacobian determinant, its inverse inverse and mass matrix (3D)
        # This maps dxyz/drst
        # Gere we store the jacobian determinant as "jac"
        self.jac = np.zeros_like(self.dxdr, dtype=np.double)
        # This maps drst/dxyz
        self.jac_inv = np.zeros_like(self.dxdr, dtype=np.double)
        self.B = np.zeros_like(self.dxdr, dtype=np.double)

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
                            * np.diag(self.w3).reshape((msh.lz, msh.ly, msh.lx))[
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
        if msh.gdim > 2:
            d1 = np.zeros((3), dtype=np.double)
            d2 = np.zeros((3), dtype=np.double)
            self.area = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=np.double)
            self.nx = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=np.double)
            self.ny = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=np.double)
            self.nz = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=np.double)

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

    def dudrst(self, field, dr):
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

        dudrst = np.zeros_like(field, dtype=np.double)

        for e in range(0, nelv):
            tmp = field[e, :, :, :].reshape(-1, 1)
            dtmp = dr @ tmp
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
        dudxyz = np.zeros_like(field, dtype=np.double)

        dfdr = self.dudrst(field, self.dr)
        dfds = self.dudrst(field, self.ds)
        if self.gdim > 2:
            dfdt = self.dudrst(field, self.dt)

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
            tmp = np.copy(field)
            for ind in range(0, len(msh.nonlinear_shared_points)):
                ind1 = np.array(msh.nonlinear_indices[ind])
                ind2 = np.array(msh.nonlinear_shared_points[ind])
                field[ind1[:, 0], ind1[:, 1], ind1[:, 2], ind1[:, 3]] = np.mean(
                    tmp[ind2[:, 0], ind2[:, 1], ind2[:, 2], ind2[:, 3]]
                )
        else:
            print("Mesh does not have connectivity data. Returning unmodified array")

        return field


# -----------------------------------------------------------------------


## Define functions for the calculation of the quadrature points (Taken from the lecture notes)
def GLC_pwts(n):
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

    x = np.cos(np.arange(n) * pi / (n - 1))
    w = np.zeros(n)
    for i in range(n):
        tmp_ = 0.0
        for k in range(int((n - 1) / 2)):
            tmp_ += delt(2 * k, n) / (1 - 4.0 * k**2) * np.cos(2 * i * pi * k / (n - 1))
        w[i] = tmp_ * delt(i, n) * 4 / float(n - 1)
    return x, w


def GLL_pwts(n, eps=10**-8, max_iter=1000):
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
    V = np.zeros((n, n))  # Legendre Vandermonde Matrix
    # Initial guess for the nodes: GLC points
    xi, _ = GLC_pwts(n)
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


def get_transform_matrix(n, dim):
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
        n
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x = np.flip(x)
    w = np.flip(w_)

    # Create a diagonal matrix
    ww = np.eye(n)
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
    leg = np.zeros((p, p))
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
    delta = np.ones(n)
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
    v2d = np.kron(v, v)
    vinv = leg.T @ ww
    vinv2d = np.kron(vinv, vinv)

    # 3d transformation matrix
    v = leg
    v3d = np.kron(v, np.kron(v, v))
    vinv = leg.T @ ww
    vinv3d = np.kron(vinv, np.kron(vinv, vinv))

    if dim == 1:
        vv = v
        vvinv = vinv
        w3 = w
    elif dim == 2:
        vv = v2d
        vvinv = vinv2d
        w3 = np.kron(ww, ww)
    else:
        vv = v3d
        vvinv = vinv3d
        w3 = np.kron(ww, np.kron(ww, ww))

    return vv, vvinv, w3, x, w


def get_derivative_matrix(n, dim):
    """
    Derivative matrix of Lagrange polynomials a GLL points.

    Parameters
    ----------
    n : int
        Polynomial degree (order - 1).

    dim : int
        Dimension of the problem.

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
        n
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x = np.flip(x)
    w = np.flip(w_)

    # Create a diagonal matrix
    ww = np.eye(n)
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
    leg = np.zeros((p, p))
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

    d_n = np.zeros((p, p))

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
        dx2d = np.kron(np.eye(p), d_n)
        dy2d = np.kron(d_n, np.eye(p))

        dx = dx2d
        dy = dy2d
        dz = None
    else:
        dx3d = np.kron(np.eye(p), np.kron(np.eye(p), d_n))
        dy3d = np.kron(np.eye(p), np.kron(d_n, np.eye(p)))
        dz3d = np.kron(d_n, np.kron(np.eye(p), np.eye(p)))

        dx = dx3d
        dy = dy3d
        dz = dz3d

    return dx, dy, dz, d_n
