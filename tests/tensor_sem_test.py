import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Import general modules
import numpy as np

# Import relevant modules
from pynektools.interpolation.mesh_to_mesh import p_refiner_c
from pynektools.interpolation.sem import element_interpolator_c
from pynektools.ppymech.neksuite import preadnek
from pynektools.datatypes.msh import msh_c
from pynektools.datatypes.coef import coef_c
from pynektools.datatypes.field import field_c

# Read the original mesh data
fname = '../examples/data/rbc0.f00001'
data     = preadnek(fname, comm)
msh      = msh_c(comm, data = data)
del data

# Create a refined mesh
pref = p_refiner_c(n_old = msh.lx, n_new = 10)
msh_ref = pref.get_new_mesh(comm, msh = msh)

# Create the element interpolator for both meshes
ei = element_interpolator_c(msh.lx)
ei_ref = element_interpolator_c(msh_ref.lx)

if comm.Get_rank() == 0:

    # ======================================================#
    # Test interpolating all points within the same element #
    # ======================================================#

    # Now choose 1 element to interpolate
    elem = 1

    # Get the extact rst coordinates of the refined element
    exact_r = ei_ref.x_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_s = ei_ref.y_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_t = ei_ref.z_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 

    # Allocate found coordinates
    found_r = np.zeros_like(exact_r)
    found_s = np.zeros_like(exact_s)
    found_t = np.zeros_like(exact_s)
    
    found_x = np.zeros_like(exact_r)
    found_y = np.zeros_like(exact_s)
    found_z = np.zeros_like(exact_s)
    
    # Find the rst coordinates of the refined element
    start_time = MPI.Wtime()
    for k in range(0, msh_ref.lz):
        for j in range(0, msh_ref.ly):
            for i in range(0, msh_ref.lx):
                ei.project_element_into_basis(msh.x[elem,:,:,:], msh.y[elem,:,:,:], msh.z[elem,:,:,:])    
                found_r[k,j,i], found_s[k,j,i], found_t[k,j,i] = ei.find_rst_from_xyz(msh_ref.x[elem,k,j,i], msh_ref.y[elem,k,j,i], msh_ref.z[elem,k,j,i])
    print('sem: Time to find rst: {}'.format(MPI.Wtime() - start_time))
    print('sem: The last point took: {} iterations'.format(ei.iterations))

    print('sem: The r coordinates were found correctly: {}'.format(np.allclose(exact_r, found_r)))
    print('sem: The s coordinates were found correctly: {}'.format(np.allclose(exact_s, found_s)))
    print('sem: The t coordinates were found correctly: {}'.format(np.allclose(exact_t, found_t)))
    
    # Interpolate back to the xyz coordinates
    for k in range(0, msh_ref.lz):
        for j in range(0, msh_ref.ly):
            for i in range(0, msh_ref.lx):
                found_x[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.x[elem,:,:,:])
                found_y[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.y[elem,:,:,:])
                found_z[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.z[elem,:,:,:])
    

    print('sem: The x coordinates were found correctly: {}'.format(np.allclose(msh_ref.x[elem,:,:,:], found_x)))
    print('sem: The y coordinates were found correctly: {}'.format(np.allclose(msh_ref.y[elem,:,:,:], found_y)))
    print('sem: The z coordinates were found correctly: {}'.format(np.allclose(msh_ref.z[elem,:,:,:], found_z)))
    
    
    # ========================================================================================#
    # Test interpolating all points within the same element using the tensor supported module #
    # ========================================================================================#
    print('=====================')

    from pynektools.interpolation.tensor_sem import element_interpolator_c as tensor_element_interpolator_c

    # Instance the tensor interpolator
    max_pts   = 128
    max_elems = 1
    tei       = tensor_element_interpolator_c(msh.lx, max_pts=max_pts)
     
    # Allocate the data with the candidate elements per point
    rshp_x = np.zeros((max_pts, 1, msh.lz, msh.ly, msh.lx))
    rshp_y = np.zeros((max_pts, 1, msh.lz, msh.ly, msh.lx))
    rshp_z = np.zeros((max_pts, 1, msh.lz, msh.ly, msh.lx))

    # Project all the involved elements at the same time
    rshp_x[:,:,:,:] = msh.x[elem,:,:,:].reshape(1, 1, msh.lz, msh.ly, msh.lx)
    rshp_y[:,:,:,:] = msh.y[elem,:,:,:].reshape(1, 1, msh.lz, msh.ly, msh.lx)
    rshp_z[:,:,:,:] = msh.z[elem,:,:,:].reshape(1, 1, msh.lz, msh.ly, msh.lx)
    tei.project_element_into_basis(rshp_x, rshp_y, rshp_z)

    # Get the projection from the non tensor interpolator to compare
    ei.project_element_into_basis(msh.x[elem,:,:,:], msh.y[elem,:,:,:], msh.z[elem,:,:,:])

    print('tensor_sem: The xhat is projected correctly: {}'.format(np.allclose(ei.x_e_hat, tei.x_e_hat[0,0,:,:])))
    print('tensor_sem: The yhat is projected correctly: {}'.format(np.allclose(ei.y_e_hat, tei.y_e_hat[0,0,:,:])))
    print('tensor_sem: The zhat is projected correctly: {}'.format(np.allclose(ei.z_e_hat, tei.z_e_hat[0,0,:,:])))

    # Reshape the points to the new format
    new_r = found_r.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
    new_s = found_s.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
    new_t = found_t.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
    new_x = found_x.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
    new_y = found_y.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
    new_z = found_z.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)

    # Get the xyz coordinates from the rst coordinates with the tensor
    xx = np.zeros_like(new_x)
    yy = np.zeros_like(new_y)
    zz = np.zeros_like(new_z)

    total_pts = msh_ref.lz*msh_ref.ly*msh_ref.lx
    total_iterations = np.ceil(total_pts / max_pts)

    start_time = MPI.Wtime()
    for i in range(0, int(total_iterations)):
        start = i * max_pts
        end   = (i+1) * max_pts
        if end > total_pts:
            end = total_pts
        npoints = end - start
        tei.project_element_into_basis(rshp_x[:npoints,:,:,:], rshp_y[:npoints,:,:,:], rshp_z[:npoints,:,:,:])
        xx[start:end], yy[start:end], zz[start:end] = tei.get_xyz_from_rst(new_r[start:end], new_s[start:end], new_t[start:end], apply_1d_ops = True)
    print('tensor_sem: Time to get all xyz: {}'.format(MPI.Wtime() - start_time))
 
    print('tensor_sem: The x coordinates were calculated corrently from rst: {}'.format(np.allclose(xx, new_x)))
    print('tensor_sem: The y coordinates were calculated corrently from rst: {}'.format(np.allclose(yy, new_y)))
    print('tensor_sem: The z coordinates were calculated corrently from rst: {}'.format(np.allclose(zz, new_z)))

    # Check the gradients
    if max_pts == 1:
        x1, y1, z1 = ei.get_xyz_from_rst(-0.3, 0.3, 0.4, apply_1d_ops = True)
        test_r = (np.ones((1))*-0.3).reshape((1, 1, 1, 1))
        test_s = (np.ones((1))*0.3).reshape((1, 1, 1, 1))
        test_t = (np.ones((1))*0.4).reshape((1, 1, 1, 1))
        x2, y2, z2 = tei.get_xyz_from_rst(test_r, test_s, test_t)
        print('tensor_sem: Jacobian calculated correctly: {}'.format(np.allclose(ei.jac, tei.jac[0,0,:,:])))


    print('=====================')
    # Get the xyz coordinates from the rst coordinates with the tensor
    rr = np.zeros_like(new_r)
    ss = np.zeros_like(new_s)
    tt = np.zeros_like(new_t)

    total_pts = msh_ref.lz*msh_ref.ly*msh_ref.lx
    total_iterations = np.ceil(total_pts / max_pts)

    #Check the search function
    start_time = MPI.Wtime()
    for i in range(0, int(total_iterations)):
        start = i * max_pts
        end   = (i+1) * max_pts
        if end > total_pts:
            end = total_pts
        npoints = end - start
        tei.project_element_into_basis(rshp_x[:npoints,:,:,:], rshp_y[:npoints,:,:,:], rshp_z[:npoints,:,:,:])
        rr[start:end], ss[start:end], tt[start:end] = tei.find_rst_from_xyz(new_x[start:end], new_y[start:end], new_z[start:end])
    print('tensor_sem: Time to find rst: {}'.format(MPI.Wtime() - start_time))
    print('tensor_sem: Last run took: {} iterations'.format(ei.iterations))
    
    print('tensor_sem: The r coordinates were found correctly: {}'.format(np.allclose(rr, new_r)))
    print('tensor_sem: The s coordinates were found correctly: {}'.format(np.allclose(ss, new_s)))
    print('tensor_sem: The t coordinates were found correctly: {}'.format(np.allclose(tt, new_t)))
