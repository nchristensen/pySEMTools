from mpi4py import MPI #equivalent to the use of MPI_init() in C
comm = MPI.COMM_WORLD

from pynektools.monitoring.memory_monitor import MemoryMonitor

mm = MemoryMonitor()

from pynektools.io.ppymech.neksuite import pynekread, pynekwrite
from pynektools.datatypes.msh import Mesh as msh_c
from pynektools.datatypes.field import Field as field_c
from pynektools.datatypes.coef import Coef as coef_c
import numpy as np

mm.system_memory_usage(comm, "pynektools_imports")

fname = "../data/rbc0.f00001"
ddtype = np.single
create_connectivity = False
init_coef = True
get_area = False

#@profile
def test(fname, comm): 
    
    msh = msh_c(comm, create_connectivity=create_connectivity)
    fld = field_c(comm)
    start_time = MPI.Wtime()
    pynekread(fname, comm, msh = msh, fld = fld, data_dtype=ddtype) 
    if comm.Get_rank() == 0:
        print(f"Time to read the data: {MPI.Wtime() - start_time} s")

    mm.object_memory_usage(comm, msh, "msh", print_msg=False)
    mm.object_memory_usage_per_attribute(comm, msh, "msh", print_msg=False)
    mm.object_memory_usage(comm, fld, "fld", print_msg=False)
    mm.object_memory_usage_per_attribute(comm, fld, "fld", print_msg=False)

    if init_coef:

        start_time = MPI.Wtime()
        coef = coef_c(msh, comm, get_area=get_area)
        if comm.Get_rank() == 0:
            print(f"Time to create the coef object: {MPI.Wtime() - start_time} s")
    
        mm.object_memory_usage(comm, coef, "coef", print_msg=False)
        mm.object_memory_usage_per_attribute(comm, coef, "coef", print_msg=False)
         
    if comm.Get_rank() == 0:
        for key in mm.object_report.keys():
            mm.report_object_information(comm, key)
            print('===================================')
            print('===================================')
            print('===================================')

    mm.system_memory_usage(comm, "Finished_execution")
    
    return

test(fname, comm)

# Uncomment this if you want to see how the system memory usage changes over time
if comm.Get_rank() == 0:
    mm.report_system_information(comm)