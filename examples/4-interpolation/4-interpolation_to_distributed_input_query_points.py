# In this example, we interpolate the same points in every rank, so this should not be done in practice
# A proper for this, would have different points in each rank

# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np

# Get mpi info
comm = MPI.COMM_WORLD

from pynektools.io.ppymech.neksuite import pynekread
from pynektools.datatypes.msh import Mesh
from pynektools.datatypes.field import FieldRegistry

msh = Mesh(comm, create_connectivity=True)
fld = FieldRegistry(comm)
fname = '../data/rbc0.f00001'
pynekread(fname, comm, data_dtype=np.double, msh=msh, fld=fld)

# Import helper functions
import pynektools.interpolation.utils as interp_utils
import pynektools.interpolation.pointclouds as pcs

#if comm.Get_rank() == 0 :
if 0 == 0:
    # Generate the bounding box of the points
    x_bbox = [0, 0.05]
    y_bbox = [0, 2*np.pi]
    z_bbox = [0 , 1]
    nx = 30
    ny = 30
    nz = 30
    
    # Generate the 1D mesh
    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")

    # Generate a 3D mesh
    r, th, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x = r*np.cos(th)
    y = r*np.sin(th)

    # Array the points as a list of probes
    xyz = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])

    # Write the points for future use
    with open('points.csv', 'w') as f:
        for i in range((xyz.shape[0])):
            f.write(f"{xyz[i][0]},{xyz[i][1]},{xyz[i][2]}\n")
else:
    xyz = None

from pynektools.interpolation.probes import Probes

probes = Probes(comm, probes = xyz, msh = msh, point_interpolator_type='multiple_point_legendre_numpy', max_pts=256, find_points_comm_pattern='point_to_point')

probes.interpolate_from_field_list(0, [fld.registry['w']], comm, write_data=True)