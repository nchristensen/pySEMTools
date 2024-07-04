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

# This part onwards is form find_rst

not_found_code = -10
elem_percent_expansion = 0.01
offset_el = 0
max_elems = 1
rank  = comm.Get_rank()
use_test_pattern = True
use_torch = False

if not use_torch:
    max_pts = 128
else:
    max_pts = itp.probe_partition.shape[0]

from pynektools.interpolation.tensor_sem import element_interpolator_c as tensor_element_interpolator_c
tei = tensor_element_interpolator_c(t_itp.ei.n, max_pts=max_pts, use_torch = use_torch)

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


# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================

# Start by finding multiple points with 1 element at a time

## Set up a buffer to hold the data
if not use_torch:
    r = np.zeros((max_pts, max_elems, 1, 1), dtype = np.double)
    s = np.zeros((max_pts, max_elems, 1, 1), dtype = np.double)
    t = np.zeros((max_pts, max_elems, 1, 1), dtype = np.double)
    test_interp = np.zeros((max_pts, max_elems, 1, 1), dtype = np.double)
else:
    r = torch.zeros((max_pts, max_elems, 1, 1), dtype=torch.float64, device=device)
    s = torch.zeros((max_pts, max_elems, 1, 1), dtype=torch.float64, device=device)
    t = torch.zeros((max_pts, max_elems, 1, 1), dtype=torch.float64, device=device)
    test_interp = torch.zeros((max_pts, max_elems, 1, 1), dtype=torch.float64, device=device)

# Identify variables
pts_n = probe.shape[0]
max_candidate_elements = np.max([len(elist) for elist in element_candidates])
iterations = np.ceil((pts_n / max_pts))
checked_elements = [[] for i in range(0, pts_n)]

start_time = MPI.Wtime()
exit_flag = False
# The following logic only works for nelems = 1
npoints = 10000
nelems = 1
for e in range(0, max_candidate_elements):   
    if exit_flag:
        break
    for j in range(0, int(iterations)):
        if npoints == 0:
            exit_flag = True
            break
 
        # Get the index of points that have not been found
        pt_not_found_indices = np.where(err_code != 1)[0]
        # Get the indices of these points that still have elements remaining to check
        pt_not_found_indices = pt_not_found_indices[np.where([ len(checked_elements[i]) < len(element_candidates[i]) for i in pt_not_found_indices])[0]]
        # Select only the maximum number of points
        pt_not_found_indices = pt_not_found_indices[:max_pts]
 
        # See which element should be checked in this iteration
        temp_candidates = [element_candidates[i] for i in pt_not_found_indices]
        temp_checked = [checked_elements[i] for i in pt_not_found_indices] 
        temp_to_check_ = [list(set(temp_candidates[i]) - set(temp_checked[i])) for i in range(len(temp_candidates))]
        # Sort them by order of closeness
        temp_to_check = [sorted(temp_to_check_[i], key=temp_candidates[i].index) for i in range(len(temp_candidates))]
        
        elem_to_check_per_point = [elist[0] for elist in temp_to_check]
        # Update the checked elements
        for i in range(0, len(pt_not_found_indices)):
            checked_elements[pt_not_found_indices[i]].append(elem_to_check_per_point[i])

        npoints = len(pt_not_found_indices)

        if npoints == 0:
            exit_flag = True
            break

        probe_new_shape = (npoints, 1, 1, 1)
        elem_new_shape = (npoints, nelems, t_itp.x.shape[1], t_itp.x.shape[2],t_itp.x.shape[3])
        
        tei.project_element_into_basis(t_itp.x[elem_to_check_per_point].reshape(elem_new_shape), t_itp.y[elem_to_check_per_point].reshape(elem_new_shape), t_itp.z[elem_to_check_per_point].reshape(elem_new_shape), use_torch=use_torch)
        r[:npoints, :nelems], s[:npoints, :nelems], t[:npoints, :nelems] = tei.find_rst_from_xyz(probe[pt_not_found_indices, 0].reshape(probe_new_shape), probe[pt_not_found_indices, 1].reshape(probe_new_shape), probe[pt_not_found_indices, 2].reshape(probe_new_shape), use_torch=use_torch)

        #Reshape results
        result_r = r[:npoints, :nelems, :, :].reshape((len(pt_not_found_indices)))
        result_s = s[:npoints, :nelems, :, :].reshape((len(pt_not_found_indices)))
        result_t = t[:npoints, :nelems, :, :].reshape((len(pt_not_found_indices)))
        result_code_bool = tei.point_inside_element[:npoints, :nelems, :, :].reshape((len(pt_not_found_indices)))
        # Assign the error codes
        if not use_torch:

            # Update indices of points that were found and those that were not
            pt_found_this_it = np.where(result_code_bool)[0]
            pt_not_found_this_it = np.where(~result_code_bool)[0]

            # Create a list with the original indices for each of this
            real_index_pt_found_this_it = [pt_not_found_indices[pt_found_this_it[i]] for i in range(0, len(pt_found_this_it))]
            real_index_pt_not_found_this_it = [pt_not_found_indices[pt_not_found_this_it[i]] for i in range(0, len(pt_not_found_this_it))]

            # Update codes for points found in this iteration
            probe_rst[real_index_pt_found_this_it, 0] = result_r[pt_found_this_it]
            probe_rst[real_index_pt_found_this_it, 1] = result_s[pt_found_this_it]
            probe_rst[real_index_pt_found_this_it, 2] = result_t[pt_found_this_it]
            el_owner[real_index_pt_found_this_it] = np.array(elem_to_check_per_point)[pt_found_this_it]
            glb_el_owner[real_index_pt_found_this_it] = el_owner[real_index_pt_found_this_it] + offset_el
            rank_owner[real_index_pt_found_this_it] = rank
            err_code[real_index_pt_found_this_it] = 1

            # If user has selected to check a test pattern:
            if use_test_pattern: 
                
                # Get shapes
                ntest = len(pt_not_found_this_it)
                test_probe_new_shape = (ntest, nelems, 1, 1)
                test_elem_new_shape = (ntest, nelems, t_itp.x.shape[1], t_itp.x.shape[2],t_itp.x.shape[3])

                # Define new arrays (On the cpu)                
                test_elems = np.array(elem_to_check_per_point)[pt_not_found_this_it]
                test_fields = t_itp.x[test_elems,:,:,:]**2 + t_itp.y[test_elems,:,:,:]**2 + t_itp.z[test_elems,:,:,:]**2
                test_probes = probe[real_index_pt_not_found_this_it, 0]**2 +  probe[real_index_pt_not_found_this_it, 1]**2 + probe[real_index_pt_not_found_this_it, 2]**2 

                # Perform the test interpolation
                test_interp[:ntest, :nelems] = tei.interpolate_field_at_rst(result_r[pt_not_found_this_it].reshape(test_probe_new_shape), result_s[pt_not_found_this_it].reshape(test_probe_new_shape), result_t[pt_not_found_this_it].reshape(test_probe_new_shape), test_fields.reshape(test_elem_new_shape), use_torch=use_torch)
                test_result = test_interp[:ntest, :nelems].reshape(ntest)

                # Check if the test pattern is satisfied
                test_error = abs(test_probes - test_result)

                # Now assign 
                real_list = np.array(real_index_pt_not_found_this_it)
                relative_list = np.array(pt_not_found_this_it)
                better_test = np.where(test_error < test_pattern[real_index_pt_not_found_this_it])[0]

                if len(better_test) > 0:
                    probe_rst[real_list[better_test], 0] = result_r[relative_list[better_test]]
                    probe_rst[real_list[better_test], 1] = result_s[relative_list[better_test]]
                    probe_rst[real_list[better_test], 2] = result_t[relative_list[better_test]]
                    el_owner[real_list[better_test]] = np.array(elem_to_check_per_point)[relative_list[better_test]]
                    glb_el_owner[real_list[better_test]] = el_owner[real_list[better_test]] + offset_el
                    rank_owner[real_list[better_test]] = rank
                    err_code[real_list[better_test]] = not_found_code
                    test_pattern[real_list[better_test]] = test_error[better_test]

            else:
                
                probe_rst[real_index_pt_not_found_this_it, 0] = result_r[pt_not_found_this_it]
                probe_rst[real_index_pt_not_found_this_it, 1] = result_s[pt_not_found_this_it]
                probe_rst[real_index_pt_not_found_this_it, 2] = result_t[pt_not_found_this_it]
                el_owner[real_index_pt_not_found_this_it] = np.array(elem_to_check_per_point)[pt_not_found_this_it]
                glb_el_owner[real_index_pt_not_found_this_it] = el_owner[real_index_pt_not_found_this_it] + offset_el
                rank_owner[real_index_pt_not_found_this_it] = rank
                err_code[real_index_pt_not_found_this_it] = not_found_code


        else:

            result_code_bool = result_code_bool.cpu().numpy()
            result_r = result_r.cpu().numpy()
            result_s = result_s.cpu().numpy()
            result_t = result_t.cpu().numpy()

            pt_found_this_it = np.where(result_code_bool)[0]
            pt_not_found_this_it = np.where(~result_code_bool)[0]

            # Create a list with the original indices for each of this
            real_index_pt_found_this_it = [pt_not_found_indices[pt_found_this_it[i]] for i in range(0, len(pt_found_this_it))]
            real_index_pt_not_found_this_it = [pt_not_found_indices[pt_not_found_this_it[i]] for i in range(0, len(pt_not_found_this_it))]

            # Update codes for points found in this iteration
            probe_rst[real_index_pt_found_this_it, 0] = result_r[pt_found_this_it]
            probe_rst[real_index_pt_found_this_it, 1] = result_s[pt_found_this_it]
            probe_rst[real_index_pt_found_this_it, 2] = result_t[pt_found_this_it]
            el_owner[real_index_pt_found_this_it] = np.array(elem_to_check_per_point)[pt_found_this_it]
            glb_el_owner[real_index_pt_found_this_it] = el_owner[real_index_pt_found_this_it] + offset_el
            rank_owner[real_index_pt_found_this_it] = rank
            err_code[real_index_pt_found_this_it] = 1

            # If user has selected to check a test pattern:
            if use_test_pattern: 
                
                # Get shapes
                ntest = len(pt_not_found_this_it)
                test_probe_new_shape = (ntest, nelems, 1, 1)
                test_elem_new_shape = (ntest, nelems, t_itp.x.shape[1], t_itp.x.shape[2],t_itp.x.shape[3])

                # Define new arrays (On the cpu)                
                test_elems = np.array(elem_to_check_per_point)[pt_not_found_this_it]
                test_fields = t_itp.x[test_elems,:,:,:]**2 + t_itp.y[test_elems,:,:,:]**2 + t_itp.z[test_elems,:,:,:]**2
                test_probes = probe[real_index_pt_not_found_this_it, 0]**2 +  probe[real_index_pt_not_found_this_it, 1]**2 + probe[real_index_pt_not_found_this_it, 2]**2 

                # Perform the test interpolation
                test_interp[:ntest, :nelems] = tei.interpolate_field_at_rst(result_r[pt_not_found_this_it].reshape(test_probe_new_shape), result_s[pt_not_found_this_it].reshape(test_probe_new_shape), result_t[pt_not_found_this_it].reshape(test_probe_new_shape), test_fields.reshape(test_elem_new_shape), use_torch=use_torch)
                test_result = test_interp[:ntest, :nelems].reshape(ntest)

                # Check if the test pattern is satisfied
                test_error = abs(test_probes - test_result.cpu().numpy())

                # Now assign 
                real_list = np.array(real_index_pt_not_found_this_it)
                relative_list = np.array(pt_not_found_this_it)
                better_test = np.where(test_error < test_pattern[real_index_pt_not_found_this_it])[0]

                if len(better_test) > 0:
                    probe_rst[real_list[better_test], 0] = result_r[relative_list[better_test]]
                    probe_rst[real_list[better_test], 1] = result_s[relative_list[better_test]]
                    probe_rst[real_list[better_test], 2] = result_t[relative_list[better_test]]
                    el_owner[real_list[better_test]] = np.array(elem_to_check_per_point)[relative_list[better_test]]
                    glb_el_owner[real_list[better_test]] = el_owner[real_list[better_test]] + offset_el
                    rank_owner[real_list[better_test]] = rank
                    err_code[real_list[better_test]] = not_found_code
                    test_pattern[real_list[better_test]] = test_error[better_test]
                
            else: 
                
                probe_rst[real_index_pt_not_found_this_it, 0] = result_r[pt_not_found_this_it]
                probe_rst[real_index_pt_not_found_this_it, 1] = result_s[pt_not_found_this_it]
                probe_rst[real_index_pt_not_found_this_it, 2] = result_t[pt_not_found_this_it]
                el_owner[real_index_pt_not_found_this_it] = np.array(elem_to_check_per_point)[pt_not_found_this_it]
                glb_el_owner[real_index_pt_not_found_this_it] = el_owner[real_index_pt_not_found_this_it] + offset_el
                rank_owner[real_index_pt_not_found_this_it] = rank
                err_code[real_index_pt_not_found_this_it] = not_found_code



print(r.device)
print('tensor_interpolator : Time to find: {}'.format(MPI.Wtime() - start_time))

print(np.where(el_owner != itp.el_owner_partition)[0])

# Print if the results are the same
if interpolate_sem_mesh:
    print(np.allclose(probe_rst, itp.probe_rst_partition))
    print(np.allclose(el_owner, itp.el_owner_partition))

