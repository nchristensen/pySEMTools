""" Module that contains msh class, which contains relevant data on the domain"""

import numpy as np
from scipy.spatial import KDTree

NoneType = type(None)


class Mesh:
    """
    Class that contains coordinate and partitioning data of the domain.

    This class needs to be used generaly as it contains the coordinates of the domain and
    some information about the partitioning of the domain.

    Parameters
    ----------
    comm : Comm
        MPI comminicator object.
    data : HexaData, optional
        HexaData object that contains the coordinates of the domain.
    x : ndarray, optional
        X coordinates of the domain. shape is (nelv, lz, ly, lx).
    y : ndarray, optional
        Y coordinates of the domain. shape is (nelv, lz, ly, lx).
    z : ndarray, optional
        Z coordinates of the domain. shape is (nelv, lz, ly, lx).
    create_connectivity : bool, optional
        If True, the connectivity of the domain will be created. (Memory intensive).

    Attributes
    ----------
    x : ndarray
        X coordinates of the domain. shape is (nelv, lz, ly, lx).
    y : ndarray
        Y coordinates of the domain. shape is (nelv, lz, ly, lx).
    z : ndarray
        Z coordinates of the domain. shape is (nelv, lz, ly, lx).
    lx : int
        Polynomial degree in x direction.
    ly : int
        Polynomial degree in y direction.
    lz : int
        Polynomial degree in z direction.
    nelv : int
        Number of elements in the domain in current rank.
    glb_nelv : int
        Total number of elements in the domain.
    gdim : int
        Dimension of the domain.
    non_linear_shared_points : list, optional
        List that show the index where the points in the domain are shared, used by coef in dssum.

    Returns
    -------

    Examples
    --------
    If a hexadata object: data is read from disk, the mesh object can be created directly from it.

    >>> from pynektools.datatypes.msh import Mesh
    >>> msh = Mesh(comm, data = data)

    If the coordinates are already available, the mesh object can be created from them.

    >>> from pynektools.datatypes.msh import Mesh
    >>> msh = Mesh(comm, x = x, y = y, z = z)

    This is useful in situations where the coordinates are generated in the code or streamed into python from another source.
    """

    def __init__(
        self, comm, data=None, x=None, y=None, z=None, create_connectivity=True
    ):

        if not isinstance(data, NoneType):
            self.x, self.y, self.z = get_coordinates_from_hexadata(data)
        else:
            self.x = x
            self.y = y
            self.z = z

        self.lx = self.x.shape[
            3
        ]  # This is not an error, the x data is on the last index
        self.ly = self.x.shape[
            2
        ]  # This is not an error, the x data is on the last index
        self.lz = self.x.shape[
            1
        ]  # This is not an error, the x data is on the last index
        self.lxyz = self.lx * self.ly * self.lz
        self.nelv = self.x.shape[0]

        # Find the element offset of each rank so you can store the global element number
        nelv = self.x.shape[0]
        sendbuf = np.ones((1), np.intc) * nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - nelv

        # Find the total number of elements
        sendbuf = np.ones((1), np.intc) * self.nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Allreduce(sendbuf, recvbuf)
        self.glb_nelv = recvbuf[0]

        if self.lz > 1:
            self.gdim = 3
        else:
            self.gdim = 2

        self.create_connectivity = create_connectivity

        if create_connectivity:
            if self.lz > 1:
                z_ind = [0, self.lz - 1]
            else:
                z_ind = [0]
            self.coord_hash_to_shared_map = dict()
            for e in range(0, self.nelv):

                # loop through all faces (3 loops required)

                for k in z_ind:
                    for j in range(0, self.ly):
                        for i in range(0, self.lx):
                            point = (
                                self.x[e, k, j, i],
                                self.y[e, k, j, i],
                                self.z[e, k, j, i],
                            )
                            point = hash(point)
                            if point in self.coord_hash_to_shared_map:
                                # self.coord_hash_to_shared_map[point].append((e, k, j, i))
                                self.coord_hash_to_shared_map[point].append(
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                )
                            else:
                                # self.coord_hash_to_shared_map[point] = [(e, k, j, i)]
                                self.coord_hash_to_shared_map[point] = [
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                ]

                for j in [0, self.ly - 1]:
                    for k in range(self.lz):
                        for i in range(self.lx):
                            point = (
                                self.x[e, k, j, i],
                                self.y[e, k, j, i],
                                self.z[e, k, j, i],
                            )
                            point = hash(point)
                            if point in self.coord_hash_to_shared_map:
                                # self.coord_hash_to_shared_map[point].append((e, k, j, i))
                                self.coord_hash_to_shared_map[point].append(
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                )
                            else:
                                # self.coord_hash_to_shared_map[point] = [(e, k, j, i)]
                                self.coord_hash_to_shared_map[point] = [
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                ]

                for i in [0, self.lx - 1]:
                    for k in range(self.lz):
                        for j in range(self.ly):
                            point = (
                                self.x[e, k, j, i],
                                self.y[e, k, j, i],
                                self.z[e, k, j, i],
                            )
                            point = hash(point)
                            if point in self.coord_hash_to_shared_map:
                                # self.coord_hash_to_shared_map[point].append((e, k, j, i))
                                self.coord_hash_to_shared_map[point].append(
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                )
                            else:
                                # self.coord_hash_to_shared_map[point] = [(e, k, j, i)]
                                self.coord_hash_to_shared_map[point] = [
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                ]


def linear_index(i, j, k, l, lx, ly, lz):
    """
    Map 4d index to 1d.

    This is used to represent the domain as a list that can be used in search trees.

    Parameters
    ----------
    i : int
        Index in x direction.
    j : int
        Index in y direction.
    k : int
        Index in z direction.
    l : int
        Index of the element.
    lx : int
        Polynomial degree in x direction.
    ly : int
        Polynomial degree in y direction.
    lz : int
        Polynomial degree in z direction.

    Returns
    -------
    int
        1d index of the 4d index.
    """
    return i + lx * ((j - 0) + ly * ((k - 0) + lz * ((l - 0))))


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


def get_coordinates_from_hexadata(data):
    """
    Get the coordinates from a hexadata object in mesh format.

    Used to go from a hexadata object to a ndarray that can be used for operations.

    Parameters
    ----------
    data : HexaData
        HexaData object that contains the coordinates of the domain.

    Returns
    -------
    x : ndarray
        X coordinates of the domain. shape is (nelv, lz, ly, lx).
    y : ndarray
        Y coordinates of the domain. shape is (nelv, lz, ly, lx).
    z : ndarray
        Z coordinates of the domain. shape is (nelv, lz, ly, lx).
    """

    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    x = np.zeros((nelv, lz, ly, lx), dtype=np.double)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    for e in range(0, nelv):
        x[e, :, :, :] = data.elem[e].pos[0, :, :, :]
        y[e, :, :, :] = data.elem[e].pos[1, :, :, :]
        z[e, :, :, :] = data.elem[e].pos[2, :, :, :]

    return x, y, z
