""" Module providing parallel IO routines for fld files """

import numpy as np


def fld_file_read_vector_field(fh, byte_offset, ioh):
    """Function used to read a vector field from a fld file"""

    # Associate
    fld_data_size = ioh.fld_data_size
    lx = ioh.lx
    ly = ioh.ly
    lz = ioh.lz
    nelv = ioh.nelv
    lxyz = ioh.lxyz
    gdim = ioh.gdim
    tmp_sp_vector = ioh.tmp_sp_vector
    tmp_dp_vector = ioh.tmp_dp_vector

    # Allocate
    x = np.zeros(nelv * lxyz, dtype=np.double)
    y = np.zeros(nelv * lxyz, dtype=np.double)
    z = np.zeros(nelv * lxyz, dtype=np.double)

    # Read
    if fld_data_size == 4:
        fh.Read_at_all(byte_offset, tmp_sp_vector, status=None)
        i = 0
        for e in range(0, nelv):
            for j in range(0, lxyz):
                x[e * lxyz + j] = tmp_sp_vector[i]
                i += 1
            for j in range(0, lxyz):
                y[e * lxyz + j] = tmp_sp_vector[i]
                i += 1
            if gdim > 2:
                for j in range(0, lxyz):
                    z[e * lxyz + j] = tmp_sp_vector[i]
                    i += 1
    else:
        fh.Read_at_all(byte_offset, tmp_dp_vector, status=None)
        i = 0
        for e in range(0, nelv):
            for j in range(0, lxyz):
                x[e * lxyz + j] = tmp_dp_vector[i]
                i += 1
            for j in range(0, lxyz):
                y[e * lxyz + j] = tmp_dp_vector[i]
                i += 1
            if gdim > 2:
                for j in range(0, lxyz):
                    z[e * lxyz + j] = tmp_dp_vector[i]
                    i += 1

    # Reshape to pymech compatible
    x = x.reshape((nelv, lz, ly, lx))
    y = y.reshape((nelv, lz, ly, lx))
    z = z.reshape((nelv, lz, ly, lx))

    return x, y, z


def fld_file_read_field(fh, byte_offset, ioh):
    """Function used to read a scalar field from a fld file"""

    # Associate
    fld_data_size = ioh.fld_data_size
    lx = ioh.lx
    ly = ioh.ly
    lz = ioh.lz
    nelv = ioh.nelv
    lxyz = ioh.lxyz
    tmp_sp_field = ioh.tmp_sp_field
    tmp_dp_field = ioh.tmp_dp_field

    # Allocate
    x = np.zeros(nelv * lxyz, dtype=np.double)

    # Read
    if fld_data_size == 4:
        fh.Read_at_all(byte_offset, tmp_sp_field, status=None)
        i = 0
        for e in range(0, nelv):
            for j in range(0, lxyz):
                x[e * lxyz + j] = tmp_sp_field[i]
                i += 1
    else:
        fh.Read_at_all(byte_offset, tmp_dp_field, status=None)
        i = 0
        for e in range(0, nelv):
            for j in range(0, lxyz):
                x[e * lxyz + j] = tmp_dp_field[i]
                i += 1

    # Reshape to pymech compatible
    x = x.reshape((nelv, lz, ly, lx))

    return x


def fld_file_write_vector_field(fh, byte_offset, x, y, z, ioh):
    """Function used to write a vector field to a fld file"""

    # Associate
    fld_data_size = ioh.fld_data_size
    nelv = ioh.nelv
    lxyz = ioh.lxyz
    gdim = ioh.gdim
    tmp_sp_vector = ioh.tmp_sp_vector
    tmp_dp_vector = ioh.tmp_dp_vector

    # Reshape to be a column
    x = x.reshape((nelv * lxyz))
    y = y.reshape((nelv * lxyz))
    z = z.reshape((nelv * lxyz))

    # Write
    if fld_data_size == 4:

        i = 0
        for e in range(0, nelv):
            for j in range(0, lxyz):
                tmp_sp_vector[i] = x[e * lxyz + j]
                i += 1
            for j in range(0, lxyz):
                tmp_sp_vector[i] = y[e * lxyz + j]
                i += 1
            if gdim > 2:
                for j in range(0, lxyz):
                    tmp_sp_vector[i] = z[e * lxyz + j]
                    i += 1

        fh.Write_at_all(byte_offset, tmp_sp_vector, status=None)

    else:
        i = 0
        for e in range(0, nelv):
            for j in range(0, lxyz):
                tmp_dp_vector[i] = x[e * lxyz + j]
                i += 1
            for j in range(0, lxyz):
                tmp_dp_vector[i] = y[e * lxyz + j]
                i += 1
            if gdim > 2:
                for j in range(0, lxyz):
                    tmp_dp_vector[i] = z[e * lxyz + j]
                    i += 1

        fh.Write_at_all(byte_offset, tmp_dp_vector, status=None)

    return


def fld_file_write_field(fh, byte_offset, x, ioh):
    """Function used to write a scalar field to a fld file"""

    # Associate
    fld_data_size = ioh.fld_data_size
    nelv = ioh.nelv
    lxyz = ioh.lxyz
    tmp_sp_field = ioh.tmp_sp_field
    tmp_dp_field = ioh.tmp_dp_field

    # Reshape to single column
    x = x.reshape((nelv * lxyz))

    # Write
    if fld_data_size == 4:

        i = 0
        for e in range(0, nelv):
            for j in range(0, lxyz):
                tmp_sp_field[i] = x[e * lxyz + j]
                i += 1

        fh.Write_at_all(byte_offset, tmp_sp_field, status=None)

    else:
        i = 0
        for e in range(0, nelv):
            for j in range(0, lxyz):
                tmp_dp_field[i] = x[e * lxyz + j]
                i += 1

        fh.Write_at_all(byte_offset, tmp_dp_field, status=None)

    return


def fld_file_write_vector_metadata(fh, byte_offset, x, y, z, ioh):
    """Function used to write metadata of a vector field to a fld file"""

    # Associate
    nelv = ioh.nelv
    gdim = ioh.gdim

    buff = np.zeros(2 * gdim * nelv, dtype=np.single)

    j = 0
    for e in range(0, nelv):
        buff[j + 0] = np.min(x[e, :, :, :])
        buff[j + 1] = np.max(x[e, :, :, :])
        buff[j + 2] = np.min(y[e, :, :, :])
        buff[j + 3] = np.max(y[e, :, :, :])
        j += 4
        if gdim > 2:
            buff[j + 0] = np.min(z[e, :, :, :])
            buff[j + 1] = np.max(z[e, :, :, :])
            j += 2

    fh.Write_at_all(byte_offset, buff, status=None)

    return


def fld_file_write_metadata(fh, byte_offset, x, ioh):
    """Function used to write metadata of a scalar field to a fld file"""

    # Associate
    nelv = ioh.nelv

    buff = np.zeros(2 * nelv, dtype=np.single)

    j = 0
    for e in range(0, nelv):
        buff[j + 0] = np.min(x[e, :, :, :])
        buff[j + 1] = np.max(x[e, :, :, :])
        j += 2

    fh.Write_at_all(byte_offset, buff, status=None)

    return
