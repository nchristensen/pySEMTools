# Import general modules
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import matplotlib.pyplot as plt

# Import MPI
from mpi4py import MPI #equivalent to the use of MPI_init() in C
# Import IO helper functions
from pynektools.io.utils import io_path_data
# Import modules for reading and writing
from pynektools.ppymech.neksuite import preadnek
# Import the data types
from pynektools.datatypes.msh import msh_c
from pynektools.datatypes.field import field_c
from pynektools.datatypes.utils import write_fld_subdomain_from_list

# Split communicator for MPI - MPMD
worldcomm = MPI.COMM_WORLD
worldrank = worldcomm.Get_rank()
worldsize = worldcomm.Get_size()
col = 1
comm = worldcomm.Split(col,worldrank)
rank = comm.Get_rank()
size = comm.Get_size()

# Open input file to see path
f = open ("inputs.json", "r")
params_file = json.loads(f.read())
f.close()

# Start time
start_time = MPI.Wtime()

# Read the data paths from the input file
mesh_data = io_path_data(params_file["IO"]["mesh_data"])
field_data = io_path_data(params_file["IO"]["field_data"])
number_of_snapshots = params_file["number_of_snapshots"]

# Initialize the mesh file
path     = mesh_data.dataPath
casename = mesh_data.casename
index    = mesh_data.index
fname    = path+casename+'0.f'+str(index).zfill(5)
data     = preadnek(fname, comm)
msh      = msh_c(comm, data = data)
del data


# Loop through the snapshots
for snapshot_id in range(0, number_of_snapshots):

    # Recieve the data from fortran
    path     = field_data.dataPath
    casename = field_data.casename
    index    = field_data.index
    fname=path+casename+'0.f'+str(index + snapshot_id).zfill(5)
    if comm.Get_rank() == 0: print(" ==== Reading file: {} ======".format(fname))
    fld_data = preadnek(fname, comm)

    # Get the data in field format
    fld = field_c(comm, data = fld_data)

    u = fld.fields["vel"][0]
    v = fld.fields["vel"][1]
    w = fld.fields["vel"][2]

    # Write the data in a subdomain and with a different order than what was read
    fname="./"+"subdomain"+'0.f'+str(index + snapshot_id).zfill(5)
    write_fld_subdomain_from_list(fname, comm, msh, field_list=[u,v,w], subdomain=[[-5, 20], [-5, 5], [-1, 1]], p = 4)
