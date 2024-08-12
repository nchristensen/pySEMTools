# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
import os

# Import general modules
import numpy as np
# Import relevant modules
from pynektools.io.ppymech.neksuite import pynekread, pynekwrite
from pynektools.postprocessing.statistics.time_averaging import average_field_files
from pynektools.postprocessing.file_indexing import index_files_from_folder
from pynektools.datatypes.field import Field
from pynektools.datatypes.msh import Mesh


def test_time_averaging_1_batches():

    dtype = np.single

    # Read the mesh data
    fname = 'examples/data/rbc0.f00001'    
    # Read the original mesh data
    msh = Mesh(comm, create_connectivity=False)
    pynekread(fname, comm, data_dtype = dtype, msh = msh)

    # Create fiction mean fields
    fld = Field(comm)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 10
    u2 = np.ones_like(msh.x) * 20
    u3 = np.ones_like(msh.x) * 30
    s1 = np.ones_like(msh.x) * 40

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 20
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(0).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)
    
    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 13
    u2 = np.ones_like(msh.x) * 23
    u3 = np.ones_like(msh.x) * 33
    s1 = np.ones_like(msh.x) * 43

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 42
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(1).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 17
    u2 = np.ones_like(msh.x) * 27
    u3 = np.ones_like(msh.x) * 37
    s1 = np.ones_like(msh.x) * 47

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 70
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(2).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)


    # Now index the files:
    index_files_from_folder(comm, folder_path=".", run_start_time=0, stat_start_time = 0 )

    # Now average the fields and write them to disk
    average_field_files(comm, field_index_name = "./mean_rbc.fld_index.json", output_folder = "./", output_batch_t_len=70) 


    # Now read the files and see if they are correct
    fld2 = Field(comm)
    pynekread("batch_mean_rbc0.f00000", comm, data_dtype = dtype, fld = fld2)

    mean_u1 = fld2.fields["vel"][0]
    mean_u2 = fld2.fields["vel"][1]
    mean_u3 = fld2.fields["vel"][2]
    mean_s1 = fld2.fields["scal"][0]

    t1 = np.allclose(mean_u1, np.ones_like(msh.x) * (10*20 + 13*22 + 17*28) / 70)
    t2 = np.allclose(mean_u2, np.ones_like(msh.x) * (20*20 + 23*22 + 27*28) / 70)
    t3 = np.allclose(mean_u3, np.ones_like(msh.x) * (30*20 + 33*22 + 37*28) / 70)
    t4 = np.allclose(mean_s1, np.ones_like(msh.x) * (40*20 + 43*22 + 47*28) / 70)

    passed = np.all([t1, t2, t3, t4])

    if comm.Get_rank() == 0:    
        # delete the wirtten files
        os.system("rm ./mean_rbc0.f00000")
        os.system("rm ./mean_rbc0.f00001")
        os.system("rm ./mean_rbc0.f00002")
        os.system("rm ./mean_rbc.fld_index.json")
        os.system("rm ./batches.json")
        os.system("rm ./batch_mean_rbc0.f00000")
    comm.Barrier()

    print(passed)

    assert passed

def test_time_averaging_2_batches():

    dtype = np.single

    # Read the mesh data
    fname = 'examples/data/rbc0.f00001'    
    # Read the original mesh data
    msh = Mesh(comm, create_connectivity=False)
    pynekread(fname, comm, data_dtype = dtype, msh = msh)

    # Create fiction mean fields
    fld = Field(comm)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 10
    u2 = np.ones_like(msh.x) * 20
    u3 = np.ones_like(msh.x) * 30
    s1 = np.ones_like(msh.x) * 40

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 20
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(0).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)
    
    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 13
    u2 = np.ones_like(msh.x) * 23
    u3 = np.ones_like(msh.x) * 33
    s1 = np.ones_like(msh.x) * 43

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 42
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(1).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 17
    u2 = np.ones_like(msh.x) * 27
    u3 = np.ones_like(msh.x) * 37
    s1 = np.ones_like(msh.x) * 47

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 70
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(2).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)


    # Now index the files:
    index_files_from_folder(comm, folder_path=".", run_start_time=0, stat_start_time = 0 )

    # Now average the fields and write them to disk
    average_field_files(comm, field_index_name = "./mean_rbc.fld_index.json", output_folder = "./", output_batch_t_len=42) 


    # Now read the files and see if they are correct
    fld2 = Field(comm)
    pynekread("batch_mean_rbc0.f00000", comm, data_dtype = dtype, fld = fld2)

    mean_u1 = fld2.fields["vel"][0]
    mean_u2 = fld2.fields["vel"][1]
    mean_u3 = fld2.fields["vel"][2]
    mean_s1 = fld2.fields["scal"][0]

    t1 = np.allclose(mean_u1, np.ones_like(msh.x) * (10*20 + 13*22) / 42)
    t2 = np.allclose(mean_u2, np.ones_like(msh.x) * (20*20 + 23*22) / 42)
    t3 = np.allclose(mean_u3, np.ones_like(msh.x) * (30*20 + 33*22) / 42)
    t4 = np.allclose(mean_s1, np.ones_like(msh.x) * (40*20 + 43*22) / 42)

    passed1 = np.all([t1, t2, t3, t4])
    
    fld3 = Field(comm)
    pynekread("batch_mean_rbc0.f00001", comm, data_dtype = dtype, fld = fld3)
    
    mean_u1 = fld3.fields["vel"][0]
    mean_u2 = fld3.fields["vel"][1]
    mean_u3 = fld3.fields["vel"][2]
    mean_s1 = fld3.fields["scal"][0]

    t1 = np.allclose(mean_u1, np.ones_like(msh.x) * (17*28) / 28)
    t2 = np.allclose(mean_u2, np.ones_like(msh.x) * (27*28) / 28)
    t3 = np.allclose(mean_u3, np.ones_like(msh.x) * (37*28) / 28)
    t4 = np.allclose(mean_s1, np.ones_like(msh.x) * (47*28) / 28)
    
    passed2 = np.all([t1, t2, t3, t4])

    passed = np.all([passed1, passed2])

    # delete the wirtten files
    if comm.Get_rank() == 0:
        os.system("rm ./mean_rbc0.f00000")
        os.system("rm ./mean_rbc0.f00001")
        os.system("rm ./mean_rbc0.f00002")
        os.system("rm ./mean_rbc.fld_index.json")
        os.system("rm ./batches.json")
        os.system("rm ./batch_mean_rbc0.f00000")
        os.system("rm ./batch_mean_rbc0.f00001")
    comm.Barrier()

    print(passed)

    assert passed

test_time_averaging_1_batches()
test_time_averaging_2_batches()
