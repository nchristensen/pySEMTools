from mpi4py import MPI
from pynektools.io.compress import write_field, read_field
from pynektools.datatypes.msh import Mesh
from pynektools.datatypes.field import Field
from pynektools.ppymech.neksuite import preadnek
import numpy as np

comm = MPI.COMM_WORLD

# Read the data
fname = "../data/rbc0.f00001"
data = preadnek(fname, comm)
msh = Mesh(comm, data = data)
fld = Field(comm, data = data)
del data

# Settings for the writer
fname = "compressed_rbc0.f00001"
wrd_size = 4

write_field(comm, msh=msh, fld=fld, fname=fname, wrd_size=wrd_size, write_mesh=True)

comm.Barrier()

msh2, fld2 = read_field(comm, fname=fname)

print(np.allclose(msh.x, msh2.x))
print(np.allclose(fld.fields['temp'][0], fld2.fields['temp'][0]))