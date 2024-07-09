""" Define some MPI operation wrappers"""

import numpy as np


def gather_in_root(sendbuf, root, dtype, comm):
    """Gather data from all processes to the root process"""

    rank = comm.Get_rank()

    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.allgather(sendbuf.size))

    if rank == root:
        # print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=dtype)
    else:
        recvbuf = None

    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)

    return recvbuf, sendcounts


def scatter_from_root(sendbuf, sendcounts, root, dtype, comm):
    """Scatter data from the root process to all processes"""

    rank = comm.Get_rank()

    recvbuf = np.ones(sendcounts[rank], dtype=dtype) * -100

    comm.Scatterv(sendbuf=(sendbuf, sendcounts), recvbuf=recvbuf, root=root)

    return recvbuf
