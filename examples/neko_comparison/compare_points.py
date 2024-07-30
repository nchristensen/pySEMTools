# Import the data types
from mpi4py import MPI #equivalent to the use of MPI_init() in C
from pynektools.io.ppymech.neksuite import preadnek, pwritenek
from pynektools.datatypes.msh import Mesh
from pynektools.datatypes.coef import Coef
from pynektools.datatypes.field import Field
from pynektools.interpolation.probes import Probes
import pynektools.interpolation.utils as interp_utils
import pynektools.interpolation.pointclouds as pcs
from pynektools.io.read_probes import ProbesReader
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nekofname = 'output.csv'
pythonfname = 'interpolated_fields_python.csv'


print('Reading neko outputs')
neko  = ProbesReader(nekofname)
print('Reading pynektools outputs')
pynek = ProbesReader(pythonfname)


print(np.allclose(neko.points, pynek.points, atol=1e-07))

print(neko.fields["w"].shape)
print(pynek.fields["vel2"].shape)

print(np.allclose(neko.fields["w"], pynek.fields["vel2"], atol=1e-07))
