from mpi4py import MPI
from pynektools.comm.router import Router
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Initialize router
rt = Router(comm)

# ==========================================================================
# Example on communicating data between ranks, processing and recieving back
# ==========================================================================

local_data = np.zeros(((rank+1)*10, 3), dtype=np.double)

destination = [rank + 1, rank + 2]
for i, dest in enumerate(destination):
    if dest >= size:
        destination[i] = dest - size

# First send the data to the other ranks.
# The data input in this case is not a list, therefore the same data is sent to all destinations 
sources, recvbf = rt.send_recv(destination = destination, data = local_data, dtype=np.double, tag = 0)
#reshape the data
for i in range(0, len(recvbf)):
    recvbf[i] = recvbf[i].reshape((-1, 3))

# Now modify the data
for data in recvbf:
    data[:] = rank

# Now send the data back to the original rank
# Now the destinations are the previous sources
sources, recvbf = rt.send_recv(destination = sources, data = recvbf, dtype=np.double, tag = 0)
#reshape the data
for i in range(0, len(recvbf)):
    recvbf[i] = recvbf[i].reshape((-1, 3))

# Test if the recieved data is what should have been
testbuff = [local_data + destination[i] for i in range(len(destination))]

for j in range(len(sources)):
    for j in range(len(destination)):
        if sources[i] == destination[j]:
            x = (np.allclose(recvbf[i], testbuff[j]))
            print(x)
            if x == False:
                print(f"Process failed in rank: {rank}")