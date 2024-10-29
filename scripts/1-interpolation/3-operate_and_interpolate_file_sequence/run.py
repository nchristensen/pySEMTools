# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import json
# Get mpi info
comm = MPI.COMM_WORLD

from pynektools.interpolation.probes import Probes
from pynektools.io.ppymech.neksuite import pynekread
from pynektools.datatypes.field import FieldRegistry
from pynektools.datatypes.msh import Mesh
from pynektools.datatypes.coef import Coef
from pynektools.datatypes.msh_connectivity import MeshConnectivity
from memory_profiler import profile

#@profile
def main():

    # Read the inputs file
    with open('inputs.json', 'r') as f:
        inputs = json.load(f)

    query_points_fname = inputs['query_points_fname']
    sem_mesh_fname = inputs['spectral_element_mesh_fname']
    interpolated_fields_output_fname = inputs['output_sequence_fname']
    sem_dtype_str = inputs.get('spectral_element_mesh_type_in_memory', 'single')

    if sem_dtype_str == 'single':
        sem_dtype = np.single
    elif sem_dtype_str == 'double':
        sem_dtype = np.double
    else:
        raise ValueError(f"Invalid spectral element mesh data type: {sem_dtype_str}")

    field_interpolation_settings = {}
    field_interpolation_settings['input_type'] = "file_index"
    field_interpolation_settings['file_index'] = inputs["file_index_to_interpolate"]
    field_interpolation_settings['fields_to_interpolate'] = inputs["fields_to_interpolate"]

    # Interpolation settings that must have the same format as probe
    interpolation_settings = inputs.get('interpolation_settings', {})

    # Use multiple point legendre by default
    interpolation_settings['point_interpolator_type'] = interpolation_settings.get('point_interpolator_type', 'multiple_point_legendre_numpy')


    # Read the mesh directly since you will need it later
    msh = Mesh(comm, create_connectivity=False)
    pynekread(sem_mesh_fname, comm, data_dtype=sem_dtype, msh=msh)

    # Initialize the probe object
    probes = Probes(comm, probes=query_points_fname, msh = msh, output_fname=interpolated_fields_output_fname, **interpolation_settings)

    # Initialize the coef
    coef = Coef(msh, comm, get_area=True)

    # Initialize the connectivity
    gs = MeshConnectivity(comm, msh=msh)

    # Prepare a list with the inputs
    typ = field_interpolation_settings.get('input_type', None)
    if typ == "file_sequence":
        pass
    elif typ == "file_index":
        
        with open(field_interpolation_settings["file_index"], "r") as infile:
            index_file = json.load(infile)

        field_interpolation_settings['file_sequence'] = []
        for key in index_file.keys():
            try:
                int_key = int(key)
            except ValueError:
                continue
            field_interpolation_settings['file_sequence'].append(index_file[key]['path'])     
    else:
        raise ValueError("The input type must be either file_sequence or file_index")

    # Initialize a field to be used to read
    fld = FieldRegistry(comm)
    
    # Loop throught the files in the sequence
    for fname in field_interpolation_settings['file_sequence']:

        # Read the field in the sequence    
        pynekread(fname, comm, data_dtype=sem_dtype, fld=fld)

        ########################### Put operations here ###########################

        # Derivatives
        dudz = coef.dudxyz(fld.registry["u"], coef.drdz, coef.dsdz, coef.dtdz)
        dvdz = coef.dudxyz(fld.registry["v"], coef.drdz, coef.dsdz, coef.dtdz)

        # Dssum
        dudz = gs.dssum(field=dudz, msh = msh, average="multiplicity")
        dvdz = gs.dssum(field=dvdz, msh = msh, average="multiplicity")

        ########################### Put the fields you want to interpolate in the lists below ###########################        

        field_list = [dudz, dvdz]
        field_names = ["dudz", "dvdz"]
        
        ###########################################################################
 
        # Interpolate the fields
        probes.interpolate_from_field_list(fld.t, field_list, comm, write_data=True, field_names=field_names)

        # Clear the fields
        fld.clear()










if __name__ == "__main__":
    main()