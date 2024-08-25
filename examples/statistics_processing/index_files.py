from mpi4py import MPI
import os
from pynektools.postprocessing.file_indexing import index_files_from_folder

# Get communicator
comm = MPI.COMM_WORLD


# Clone the repository with the data
os.system("git clone https://github.com/adperezm/sem_data.git ../data/sem_data")

# Index the files
index_files_from_folder(comm, folder_path="../data/sem_data/statistics/channel_nelv_600/", run_start_time=0, stat_start_time = 50)
