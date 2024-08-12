import json
import numpy as np
import os

from ...monitoring.logger import Logger
from ...datatypes.msh import Mesh
from ...datatypes.field import Field
from ...io.ppymech.neksuite import preadnek, pwritenek, pynekread, pynekwrite

def average_field_files(comm, field_index_name= "", output_folder = "./", output_batch_t_len = 50, mesh_index = '', dtype = np.single):

    logger = Logger(comm=comm, module_name="average_field_files")

    logger.write("info", f"Averaging field files in index: {field_index_name}") 
    logger.write("info", f"Output files will be saved in: {output_folder}")

    # Read the json index file
    with open(field_index_name, "r") as f:
        file_index = json.load(f)

    logger.write("info", f"Files will be averaged in batches of {output_batch_t_len} time units")
    logger.write("info", f"Start dividing files into batches")
    
    batches = {}
    # Loop over the files and group them into batches
    batch_number = 0
    batches[batch_number] = {} 
    file_in_batch_number = 0
    current_batch_t_len = 0.0
    #for i in range(0, len(file_index)-1):
    for i, key in enumerate(file_index.keys()):
        try:
            int_key = int(key)
        except ValueError:
            continue

        file_t_interval = file_index[key]["time_interval"]

        # Add to the current batch
        if current_batch_t_len + file_t_interval <= output_batch_t_len * (1 + 5e-2): # Add a 5% tolerance
            create_new_batch = False
        else:
            batches[batch_number]["averaging_time"] = current_batch_t_len
            create_new_batch = True

        if create_new_batch:            
            batch_number += 1
            batches[batch_number] = {} 
            current_batch_t_len = 0.0
            file_in_batch_number = 0 

        batches[batch_number][file_in_batch_number] = file_index[key]
        current_batch_t_len += file_t_interval
        file_in_batch_number += 1

    # Check if averaging time is in the dictionary
    if batches[batch_number].get("averaging_time") is None:
        batches[batch_number]["averaging_time"] = current_batch_t_len

    logger.write("info", f"Finished dividing files into batches.  Writing batch information into {output_folder}batches.json")

    logger.write("info", f"Writing {output_folder}batches.json batch index")
    logger.tic()
    if comm.Get_rank() == 0:
        with open(output_folder + "batches.json", "w") as outfile:
            outfile.write(json.dumps(batches, indent=4))
    comm.Barrier()
    logger.toc()

    # Read the mesh
    if mesh_index == '':
        logger.write("warning", "Mesh index not provided")
        logger.write("warning", "Provide the mesh_index keywork with an index to a file that contiains the mesh")
        logger.write("warning", "we assume that the mesh index is 0")
        mesh_index = 0
    
    logger.write("info", f"Reading mesh from file {mesh_index} in {field_index_name}")
    logger.write("info", f"Reading mesh in precision:  {dtype}")

    msh_fname = file_index[str(mesh_index)]["path"]
    msh = Mesh(comm, create_connectivity=False)
    pynekread(msh_fname, comm, data_dtype = dtype, msh = msh)

    if comm.Get_rank() == 0: 
        print("=================================================================================================")
    
    for batch in batches.keys():
        
        logger.tic()
        logger.write("info", f"Averaging files in batch {batch}")
        
        batch_time = 0.0
        batch_mean_field = Field(comm)
        work_field = Field(comm)

        for i, file in enumerate(key for key in batches[batch].keys() if isinstance(key, (int))):

            if i == 0:
                out_fname = batches[batch][file]["fname"].split(".")[0][:-1] + '0.f' + str(batch).zfill(5)
                out_fname = 'batch_' + out_fname

            fname = batches[batch][file]["path"]
            file_dt = batches[batch][file]["time_interval"]
        
            logger.write("info", f"Processing file: {os.path.basename(fname)}, with time interval: {file_dt}")

            if i == 0:
                pynekread(fname, comm, data_dtype = dtype, fld = batch_mean_field)
        
                logger.write("info", f"Multiplying fields by dt = {file_dt}")

                # Multiply the fields by the time interval
                for field in batch_mean_field.fields.keys():
                    for qoi in range(0, len(batch_mean_field.fields[field])):
                        batch_mean_field.fields[field][qoi] = batch_mean_field.fields[field][qoi] * file_dt

            else:
                work_field.clear()
                pynekread(fname, comm, data_dtype = dtype, fld = work_field)
                
                logger.write("info", f"Multiplying fields by dt = {file_dt}")
                
                # Multiply the fields by the time interval
                for field in batch_mean_field.fields.keys():
                    for qoi in range(0, len(batch_mean_field.fields[field])):
                        batch_mean_field.fields[field][qoi] += work_field.fields[field][qoi] * file_dt

            batch_time += file_dt
                
            logger.write("info", f"Current averaging time of batch = {batch_time}")
        
        logger.write("info", f"Reading files in batch {batch} finished")
        logger.write("info", f"Dividing fields by total time in batch={batch_time}")

        # Divide by the total time
        for field in batch_mean_field.fields.keys():
            for qoi in range(0, len(batch_mean_field.fields[field])):
                batch_mean_field.fields[field][qoi] = batch_mean_field.fields[field][qoi] / batch_time

        # Write fields
        logger.write("info", f"Writing averaged fields in batch {batch} to file")

        # Set the time to be the cummulative time of the batch
        batch_mean_field.t = batch_time
        pynekwrite(output_folder + out_fname, comm, fld = batch_mean_field, msh = msh, wdsz=4, istep=batch)        

        
        logger.write("info", f"Processing of batch : {batch} done.")
        logger.toc()
        
        if comm.Get_rank() == 0: 
            print("=================================================================================================")
                