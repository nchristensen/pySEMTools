import numpy as np
from pymech.core import HexaData
from pymech.neksuite.field import Header
from pymech.core import HexaData
import sys
NoneType = type(None)

def create_hexadata_from_msh_fld(msh = None, fld = None, wdsz = 4,  istep = 0, time= 0, data_dtype = "float64", write_mesh = True):
    
    msh_fields  = msh.gdim
    vel_fields  = fld.vel_fields
    pres_fields = fld.pres_fields
    temp_fields = fld.temp_fields
    scal_fields = fld.scal_fields
    lx = msh.lx
    ly = msh.ly
    lz = msh.lz
    nelv = msh.nelv

    header = Header(wdsz, (lx, ly, lz), nelv, nelv, time,
               istep, fid=0, nb_files=1, nb_vars=(msh_fields, vel_fields, pres_fields, temp_fields, scal_fields))

    # Create the pymech hexadata object
    data = HexaData(header.nb_dims, header.nb_elems, header.orders, header.nb_vars, 0, dtype=data_dtype)
    data.time = header.time
    data.istep = header.istep
    data.wdsz = header.wdsz
    data.endian = sys.byteorder
  
    # Include the mesh
    data = put_coordinates_in_hexadata_from_msh(data, msh)
    
    # Include the fields
    for qoi in range(0, vel_fields):
        prefix = "vel"
        data = put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)
    
    for qoi in range(0, pres_fields):
        prefix = "pres"
        data = put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)
        
    for qoi in range(0, temp_fields):
        prefix = "temp"
        data = put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)
            
    for qoi in range(0, scal_fields):
        prefix = "scal"
        data = put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)
            
    return data


def put_coordinates_in_hexadata_from_msh(data, msh):
    
    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    for e in range(0,nelv):
        data.elem[e].pos[0,:,:,:] = msh.x[e,:,:,:]
        data.elem[e].pos[1,:,:,:] = msh.y[e,:,:,:]
        data.elem[e].pos[2,:,:,:] = msh.z[e,:,:,:]

    return data
    
def put_field_in_hexadata(data, field, prefix, qoi):

    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    if prefix == "vel":
        for e in range(0, nelv):
            data.elem[e].vel[qoi,:,:,:] = field[e,:,:,:]
    
    if prefix == "pres":
        for e in range(0, nelv):
            data.elem[e].pres[0,:,:,:] = field[e,:,:,:]
    
    if prefix == "temp":
        for e in range(0, nelv):
            data.elem[e].temp[0,:,:,:] = field[e,:,:,:]

    if prefix == "scal":
        for e in range(0, nelv):
            data.elem[e].scal[qoi,:,:,:] = field[e,:,:,:]


    return data
