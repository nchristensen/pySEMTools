#%% user specified function to define the interpolation points
def user_defined_interpolating_points():
    import numpy as np
    import pynektools.interpolation.pointclouds as pcs
    import pynektools.interpolation.utils as interp_utils

    # from mpi4py import MPI

    # # Get mpi info
    # comm = MPI.COMM_WORLD

    # Create the coordinates of the plane you want
    x_bbox = [0.0, 4.*np.pi]
    y_bbox = [-1, 1]
    z_bbox = [0, 4./3.*np.pi]  # See how here I am just setting this to be one value

    nx = 9*7
    ny = 9*7
    nz = 9*7  # I want my plane to be  in z

    print("generate interpolation points")

    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="tanh", gain=1.5)
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")
    x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

    xyz = interp_utils.transform_from_array_to_list(nx, ny, nz, [x, y, z])

    return xyz
###########################################################################################
###########################################################################################





###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
#%% the presumed complete workflow
###########################################################################################
###########################################################################################
from pynektools.postprocessing.statistics.RS_budgets import compute_and_write_additional_pstat_fields
from pynektools.postprocessing.statistics.RS_budgets import interpolate_all_stat_and_pstat_fields_onto_points

# filenames
# fname_mesh = "s01FD_3d0.f00001"
# fname_mean = "FD_3d0.f00001"
# fname_stat = "FD_3d0.f00001"
fname_mesh = "fluid_stats0.f00000"
fname_mean = "fluid_stats0.f00001"
fname_stat = "fluid_stats0.f00001"
which_dir = "./"
which_code = "neko"
nek5000_stat_type = ""

if_do_dssum_on_derivatives = False          # whether to do dssum to computed derivatives... not recommended
if_do_dssum_before_interp = True            # whether to do dssum before interpolation
if_create_boundingBox_for_interp = False    # whether to create a bounding box around the point cloud
                                            # useful for cases where there is a cluster of points somewhere


# step 1:
# the script to average the stat files in TIME goes here

# step 1.5:
# can take the average in space here: currently only if the output is still 3D
# the filenames might need to be changed after this call
# some_function_to_average in space

# step 2:
# compute the additional fields based on the 44 stat fields.
# do not do this for basic statistics
compute_and_write_additional_pstat_fields(which_dir,fname_mesh,fname_mean,fname_stat,\
                                          if_write_mesh=True,which_code=which_code,nek5000_stat_type=nek5000_stat_type, \
                                          if_do_dssum_on_derivatives=if_do_dssum_on_derivatives)

# step <3:
# define interpolation points
# WARNING HERE
# this should either be called from rank 0 only and then propoerly distributed into mpi ranks
# or each rank should generate its own points
xyz = user_defined_interpolating_points()

# step 3:
# interpolate the 44+N fields onto the interpolation points
# this function is not working at the moment
# For starters, we want this function to write the interpolated fields in MATLAB format so we can debug
interpolate_all_stat_and_pstat_fields_onto_points(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    xyz,
    which_code=which_code,
    nek5000_stat_type=nek5000_stat_type,
    if_do_dssum_before_interp=if_do_dssum_before_interp,
    if_create_boundingBox_for_interp=if_create_boundingBox_for_interp,
)

# step 3.5:
# potential script to perform averaging on the interpolated fields is called here
# this could include rotation, for example in the pipe

# step 4:
# function to compute budgets is called  here

# step 4.5:
# scrtipt to plot, etc. the budget terms is called here

# what is missing??

###########################################################################################
###########################################################################################
