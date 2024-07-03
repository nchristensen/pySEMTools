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

# NOW I SHOULD TEST THE TENSOR VERSION. 
# KEEP IN MIND THAT IN THEORY THIS SHOULD NOT AFFECT THE CODE BEFORE AND AFTER THE itp.find_points_comm_pairs(comm, communicate_candidate_pairs = True, elem_percent_expansion = 0.01)

# Instance new interpolator to mimic what would happend with the tensor one
t_itp = interpolator_c(msh.x, msh.y, msh.z, probes, comm, progress_bar = True, modal_search = True)

# Scatter the probes to all ranks
t_itp.scatter_probes_from_io_rank(0, comm)

# This part is from find points
from scipy.spatial import KDTree
from pynektools.interpolation.interpolator import get_bbox_from_coordinates, get_bbox_centroids_and_max_dist, pt_in_bbox

t_itp.my_bbox = get_bbox_from_coordinates(t_itp.x, t_itp.y, t_itp.z)
t_itp.my_bbox_centroids, t_itp.my_bbox_maxdist = get_bbox_centroids_and_max_dist(t_itp.my_bbox)
t_itp.my_tree = KDTree(t_itp.my_bbox_centroids)

#ONLY TEMPORARLY SET THIS LIKE THIS
probes = None
probes_rst = None

# this part is from find points
not_found = np.where(t_itp.err_code_partition != 1)[0]
n_not_found = not_found.size
probe = t_itp.probe_partition[not_found] 
probe_rst = t_itp.probe_rst_partition[not_found] 
el_owner = t_itp.el_owner_partition[not_found] 
glb_el_owner = t_itp.glb_el_owner_partition[not_found] 
rank_owner = t_itp.rank_owner_partition[not_found] 
err_code = t_itp.err_code_partition[not_found] 
test_pattern = t_itp.test_pattern_partition[not_found] 

# This part is form find_rst

not_found_code = -10
elem_percent_expansion = 0.01
offset_el = 0

err_code[:] = not_found_code

                
# Query the tree with the probes to reduce the bbox search
candidate_elements = t_itp.my_tree.query_ball_point(x=probe, r=t_itp.my_bbox_maxdist, p=2.0, eps=elem_percent_expansion, workers=1, return_sorted=False, return_length=False)
            
element_candidates = []
i = 0
for pt in probe:
    element_candidates.append([])        
    for e in candidate_elements[i]:
        if pt_in_bbox(pt, t_itp.my_bbox[e], rel_tol = elem_percent_expansion):
            element_candidates[i].append(e)
    i = i + 1


print(element_candidates[0:10])

# Start by finding multiple points with 1 element at a time
