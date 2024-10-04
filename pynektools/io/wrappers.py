"""Wrappers to ease IO"""

from typing import Union
import os
import h5py

from ..datatypes.msh import Mesh
from ..datatypes.field import FieldRegistry
from .ppymech.neksuite import pynekread
import numpy as np

def read_data(comm, fname: str, keys: list[str], parallel_io: bool = False, dtype = np.single):
    """
    Read data from a file and return a dictionary with the names of the files and keys

    Parameters
    ----------
    comm, MPI.Comm
        The MPI communicator
    fname : str
        The name of the file to read
    keys : list[str]
        The keys to read from the file
    parallel_io : bool, optional
        If True, read the file in parallel, by default False. This is aimed for hdf5 files, and currently it does not work if True

    Returns
    -------
    dict
        A dictionary with the keys and the data read from the file
    """

    # Check the file extension
    path = os.path.dirname(fname)
    prefix = os.path.basename(fname).split('.')[0]
    extension = os.path.basename(fname).split('.')[1]

    # Read the data
    if extension == 'hdf5':
        if parallel_io:
            raise NotImplementedError("Parallel IO is not implemented for hdf5 files")
        else:
            with h5py.File(fname, 'r') as f:
                data = {}
                for key in keys:
                    data[key] = f[key][:]
    
    elif extension[0] == 'f':
        
        data = {}
        msh = None
        fld = FieldRegistry(comm)

        # Go through the keys
        for key in keys:
            # If the mesh must be read, create a mesh object
            if key in ["x", "y", "z"]:
                # Read the mesh only once
                if msh is None:
                    msh = Mesh(comm)
                    pynekread(fname, comm, data_dtype=dtype, msh=msh)
                if key == "x":
                    data[key] = np.copy(msh.x)
                elif key == "y":
                    data[key] = np.copy(msh.y)
                elif key == "z":
                    data[key] = np.copy(msh.z)
            else:
                # Read the field
                fld.add_field(comm, field_name=key, file_type="fld", file_name=fname, file_key=key, dtype=dtype)
                data[key] = np.copy(fld.registry[key])
                fld.clear()         

    return data 
