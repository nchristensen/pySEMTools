from mpi4py import MPI
from pynektools.postprocessing.statistics.time_averaging import average_field_files

# Get communicator
comm = MPI.COMM_WORLD

# Average the field files in the index
average_field_files(comm, field_index_name = "./mean_field.fld_index.json", output_folder = "./", output_batch_t_len=50)