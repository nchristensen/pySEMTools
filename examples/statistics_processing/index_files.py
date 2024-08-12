from mpi4py import MPI
from pynektools.postprocessing.file_indexing import index_files_from_folder

# Get communicator
comm = MPI.COMM_WORLD

# Index the files
index_files_from_folder(comm, folder_path=".", run_start_time=0, stat_start_time = 50)