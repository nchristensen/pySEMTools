# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import json
# Get mpi info
comm = MPI.COMM_WORLD

from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.io.ppymech.neksuite import pynekread, pynekwrite


def main():

    # Read the inputs file
    with open('inputs.json', 'r') as f:
        inputs = json.load(f)

    sem_mesh_fname = inputs['spectral_element_mesh_fname']
    sem_mesh_fname = "../../examples/data/sem_data/instantaneous/cylinder_rbc_nelv_600/field0.f00801"
    sem_dtype_str = inputs.get('spectral_element_mesh_type_in_memory', 'single')

    if sem_dtype_str == 'single':
        sem_dtype = np.single
    elif sem_dtype_str == 'double':
        sem_dtype = np.double
    else:
        raise ValueError(f"Invalid spectral element mesh data type: {sem_dtype_str}")

    # Read the mesh
    msh = Mesh(comm)
    pynekread(sem_mesh_fname, comm, data_dtype=sem_dtype, msh=msh)
    coef = Coef(msh, comm)

    # Get the physical space change wrt reference coordinates.
    dxdr = coef.dudrst(msh.x, direction="r")
    dxds = coef.dudrst(msh.x, direction="s")
    dxdt = coef.dudrst(msh.x, direction="t")

    dydr = coef.dudrst(msh.y, direction="r")
    dyds = coef.dudrst(msh.y, direction="s")
    dydt = coef.dudrst(msh.y, direction="t")

    dzdr = coef.dudrst(msh.z, direction="r")
    dzds = coef.dudrst(msh.z, direction="s")
    dzdt = coef.dudrst(msh.z, direction="t")

    # Compute the difference between reference coordinates
    x_diff = np.zeros_like(coef.x) 
    x_diff[0] = (coef.x[1] - coef.x[0]) * 0.5 
    for i in range(1, msh.lx - 1):
        x_diff[i] = (coef.x[i + 1] - coef.x[i - 1])*0.5
    x_diff[-1] = (coef.x[-1] - coef.x[-2]) * 0.5
    r_diff = np.zeros((1, msh.lz, msh.ly, msh.lx), dtype=sem_dtype)
    s_diff = np.zeros((1, msh.lz, msh.ly, msh.lx), dtype=sem_dtype)
    t_diff = np.zeros((1, msh.lz, msh.ly, msh.lx), dtype=sem_dtype)
    for k in range(msh.lz):
        for j in range(msh.ly):
            for i in range(msh.lx):
                r_diff[0, k, j, i] = x_diff[i]
                s_diff[0, k, j, i] = x_diff[j]
                t_diff[0, k, j, i] = x_diff[k]

    # Then get the actual spacing of the mesh
    ## For dx, we get the magnitude of changes in the r, s, t directions
    dx = np.sqrt((dxdr * r_diff) ** 2 + (dxds * s_diff) ** 2 + (dxdt * t_diff) ** 2 )
    dy = np.sqrt((dydr * r_diff) ** 2 + (dyds * s_diff) ** 2 + (dydt * t_diff) ** 2 )
    dz = np.sqrt((dzdr * r_diff) ** 2 + (dzds * s_diff) ** 2 + (dzdt * t_diff) ** 2 )

    fld = FieldRegistry(comm)
    fld.add_field(comm, field_name="dx", field=dx, dtype=sem_dtype)
    fld.add_field(comm, field_name="dy", field=dy, dtype=sem_dtype)
    fld.add_field(comm, field_name="dz", field=dz, dtype=sem_dtype)
    
    pynekwrite("msh_check0.f00001", comm, msh = msh, fld=fld)

if __name__ == "__main__":
    main()