import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Import general modules
import numpy as np

# Import relevant modules
from pynektools.interpolation.mesh_to_mesh import p_refiner_c
from pynektools.interpolation.interpolator import interpolator_c
from pynektools.interpolation.sem import element_interpolator_c
from pynektools.ppymech.neksuite import preadnek
from pynektools.datatypes.msh import msh_c
from pynektools.datatypes.coef import coef_c
from pynektools.datatypes.field import field_c

from pynektools.interpolation.mpi_ops import gather_in_root, scatter_from_root

NoneType = type(None)


# Read the original mesh data
fname = '../examples/data/rbc0.f00001'
data     = preadnek(fname, comm)
msh      = msh_c(comm, data = data)
del data

# Create a refined mesh
n_new = 3
pref = p_refiner_c(n_old = msh.lx, n_new = n_new)
msh_ref = pref.get_new_mesh(comm, msh = msh)

# Instance an interpolator for the refined mesh to know the exact rst coordinates
ei_ref = element_interpolator_c(n_new)

# For an element this is true
exact_r = ei_ref.x_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
exact_s = ei_ref.y_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
exact_t = ei_ref.z_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 

interpolate_sem_mesh = False

if interpolate_sem_mesh:

    # Get the points to find
    probes = np.zeros((msh_ref.x.size, 3))
    probes_rst_exact = np.zeros((msh_ref.x.size, 3))
    point = 0
    for e in range(msh_ref.nelv):
        for k in range(0, msh_ref.lz):
            for j in range(0, msh_ref.ly):
                for i in range(0, msh_ref.lx):
                    probes[point, 0] = msh_ref.x[e, k, j, i]
                    probes[point, 1] = msh_ref.y[e, k, j, i]
                    probes[point, 2] = msh_ref.z[e, k, j, i]
                    probes_rst_exact[point, 0] = exact_r[k, j, i]
                    probes_rst_exact[point, 1] = exact_s[k, j, i]
                    probes_rst_exact[point, 2] = exact_t[k, j, i]
                    point = point + 1

else:
    import pynektools.interpolation.utils as interp_utils
    import pynektools.interpolation.pointclouds as pcs

    # Create a polar mesh
    nn = msh_ref.x.size
    nx = int(nn**(1/3))
    ny = int(nn**(1/3))
    nz = int(nn**(1/3))

    # Choose the boundaries of the interpolation mesh
    # boundaries
    x_bbox = [0, 0.05]
    y_bbox = [0, 2*np.pi]
    z_bbox = [0 , 1]

    # Generate the points in 1D
    start_time = MPI.Wtime()
    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")

    # Create 3D arrays
    r, th, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x = r*np.cos(th)
    y = r*np.sin(th)

    # Create a list with the points
    if comm.Get_rank() == 0:    
        probes = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])
    else:
        probes = 1

# Instance the interpolator
itp = interpolator_c(msh.x, msh.y, msh.z, probes, comm, progress_bar = True, modal_search = True)

# Scatter the probes to all ranks
itp.scatter_probes_from_io_rank(0, comm)

# Find the points
itp.find_points_comm_pairs(comm, communicate_candidate_pairs = True, elem_percent_expansion = 0.01)

# Peform needed redistributions
itp.gather_probes_to_io_rank(0, comm)
itp.redistribute_probes_to_owners_from_io_rank(0, comm)

# Now interpolate xyz to check if the process is correct
my_interpolated_fields = np.zeros((itp.my_probes.shape[0], 3), dtype = np.double)
if comm.Get_rank() == 0:
    interpolated_fields = np.zeros((probes.shape[0], 3), dtype = np.double)
else:
    interpolated_fields = None
# Set the time
my_interpolated_fields[:, 0] =  itp.interpolate_field_from_rst(msh.x)[:]
my_interpolated_fields[:, 1] =  itp.interpolate_field_from_rst(msh.y)[:]
my_interpolated_fields[:, 2] =  itp.interpolate_field_from_rst(msh.z)[:]

# Gather in rank zero for processing
root = 0
sendbuf = my_interpolated_fields.reshape((my_interpolated_fields.size))
recvbuf, _ = gather_in_root(sendbuf, root, np.double,  comm)
 
if type(recvbuf) != NoneType:
    tmp = recvbuf.reshape((int(recvbuf.size/(3)), 3))
    interpolated_fields[itp.sort_by_rank] = tmp
        
t1 = np.allclose(interpolated_fields, probes)

passed = np.all([t1])

if not passed:
    sys.exit('interpolator.py: find_points_comm_pairs, interpolate_field_from_rst: failed')
else:
    print('interpolator.py: find_points_comm_pairs, interpolate_field_from_rst: passed')

#========================================================

use_torch = False
if not use_torch:
    max_pts = 128
else:
    max_pts = itp.probe_partition.shape[0]

# Instance new interpolator to mimic what would happend with the tensor one
t_itp = interpolator_c(msh.x, msh.y, msh.z, probes, comm, progress_bar = True, modal_search = True, use_tensor = True, use_torch = use_torch,  max_pts = max_pts, max_elems = 1)  

# Scatter the probes to all ranks
t_itp.scatter_probes_from_io_rank(0, comm)

# Find the points
t_itp.find_points_comm_pairs(comm, communicate_candidate_pairs = True, elem_percent_expansion = 0.01)

if interpolate_sem_mesh:
    print(np.allclose(t_itp.probe_rst_partition, itp.probe_rst_partition))
    print(np.allclose(t_itp.el_owner_partition, itp.el_owner_partition))

# Peform needed redistributions
t_itp.gather_probes_to_io_rank(0, comm)
t_itp.redistribute_probes_to_owners_from_io_rank(0, comm)

# Now interpolate xyz to check if the process is correct
my_interpolated_fields = np.zeros((t_itp.my_probes.shape[0], 3), dtype = np.double)
if comm.Get_rank() == 0:
    interpolated_fields = np.zeros((probes.shape[0], 3), dtype = np.double)
else:
    interpolated_fields = None
# Set the time
start_time = MPI.Wtime()
my_interpolated_fields[:, 0] =  t_itp.interpolate_field_from_rst(msh.x)[:]
my_interpolated_fields[:, 1] =  t_itp.interpolate_field_from_rst(msh.y)[:]
my_interpolated_fields[:, 2] =  t_itp.interpolate_field_from_rst(msh.z)[:]
print('tensor_interpolator: Time to interpolate: {}'.format(MPI.Wtime() - start_time))

# Gather in rank zero for processing
root = 0
sendbuf = my_interpolated_fields.reshape((my_interpolated_fields.size))
recvbuf, _ = gather_in_root(sendbuf, root, np.double,  comm)
 
if type(recvbuf) != NoneType:
    tmp = recvbuf.reshape((int(recvbuf.size/(3)), 3))
    interpolated_fields[t_itp.sort_by_rank] = tmp
        
t1 = np.allclose(interpolated_fields, probes)

passed = np.all([t1])

if not passed:
    sys.exit('interpolator.py - tensor: find_points_comm_pairs, interpolate_field_from_rst: failed')
else:
    print('interpolator.py - tensor: find_points_comm_pairs, interpolate_field_from_rst: passed')