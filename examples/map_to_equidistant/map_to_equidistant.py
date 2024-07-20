# Import general modules
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

# Import MPI
from mpi4py import MPI #equivalent to the use of MPI_init() in C
# Import modules for reading and writing
from pynektools.io.ppymech.neksuite import preadnek
# Import the data types
from pynektools.datatypes.msh import Mesh
from pynektools.datatypes.field import Field
from pynektools.interpolation.mesh_to_mesh import PMapper
from pynektools.datatypes.utils import write_fld_file_from_list

# Split communicator for MPI - MPMD
worldcomm = MPI.COMM_WORLD
worldrank = worldcomm.Get_rank()
worldsize = worldcomm.Get_size()
col = 1
comm = worldcomm.Split(col,worldrank)
rank = comm.Get_rank()
size = comm.Get_size()

# Define the path to the data
fname  = "../data/tc_channel0.f00001"
data = preadnek(fname, comm)
msh = Mesh(comm, data=data, create_connectivity=False)
fld = Field(comm, data=data)
u = fld.fields['vel'][0]
v = fld.fields['vel'][1]
w = fld.fields['vel'][2]

# Initialize the mapper
# Here specify that it should be equal in the 3 direction
mapper = PMapper(n=msh.lx, distribution=['GLL', 'GLL', 'EQ'])

# Create the mesh with the new distribution
eq_msh = mapper.get_new_mesh(comm, msh=msh)
# Interpolate the fields now
mapped_fields = mapper.interpolate_from_field_list(comm, field_list=[u,v,w])
# Write the new mesh and fields
fname = "mappedfield0.f00001"
write_fld_file_from_list(fname, comm, eq_msh, field_list=mapped_fields)