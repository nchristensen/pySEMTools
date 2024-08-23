import mpi4py.MPI as MPI
import numpy as np

comm = MPI.COMM_WORLD

from pynektools.io.ppymech.neksuite import pynekread, pynekwrite
from pynektools.datatypes.msh import Mesh
from pynektools.datatypes.field import FieldRegistry
from pynektools.datatypes.msh_partitioning import MeshPartitioner


msh = Mesh(comm, create_connectivity=False)
fld = FieldRegistry(comm)
fname = "../data/rbc0.f00001"
pynekread(fname, comm, data_dtype=np.single, msh=msh, fld=fld)


condition1 = msh.z < 0.2
mp = MeshPartitioner(comm, msh=msh, conditions=[condition1])

partitioned_mesh = mp.create_partitioned_mesh(msh, partitioning_algorithm="load_balanced_linear", create_conectivity=False)
partitioned_field = mp.create_partitioned_field(fld, partitioning_algorithm="load_balanced_linear")

fname = "partitioned_field0.f00001"
pynekwrite(fname, comm, msh=partitioned_mesh, fld=partitioned_field, write_mesh=True)