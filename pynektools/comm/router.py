"This module contains the class router"

import numpy as np

class Router():
    def __init__(self, comm):

        self.comm = comm
        # Specifies a buffer to see how many points I send to each rank
        self.destination_count = np.zeros((comm.Get_size()), dtype = np.ulong) 
        # Specifies a buffer to see how many points I recieve from each rank   
        self.source_count = np.zeros((comm.Get_size()), dtype = np.ulong)    


    def send_recv(self, destination = None, data = None, dtype = None, tag = None):

        #===========================
        # Fill the destination count 
        #===========================

        self.destination_count[:] = 0
        # Check if the data to send is a list
        if isinstance(data, list):
            # If it is a list, match each destination with its data
            for dest_ind, dest in enumerate(destination):
                self.destination_count[dest] = data[dest_ind].size
        else:
            # If it is not a list, send the same data to all destinations
            self.destination_count[destination] = data.size
        
        #======================
        # Fill the source count 
        #======================
        self.source_count[:] = 0
        self.comm.Alltoall(sendbuf = self.destination_count, recvbuf = self.source_count)
        sources = np.where(self.source_count != 0)[0]

        #=========================
        # Allocate recieve buffers 
        #=========================
        recvbuff = [np.zeros((self.source_count[source]), dtype = dtype) for source in sources]        

        # =========================
        # Send and recieve the data
        # =========================

        ## set up recieve request
        recvreq = [self.comm.Irecv(recvbuff[source_ind], source = source, tag = tag) for source_ind, source in enumerate(sources)]

        ## Set up and complete the send request
        if isinstance(data, list):
            # If it is a list, send matching position to destination
            for dest_ind, dest in enumerate(destination):
                sendreq = self.comm.Isend(data[dest_ind].flatten(), dest = dest, tag = tag)
                sendreq.wait()
        else:
            # If it is not a list, send the same data to all destinations
            for dest_ind, dest in enumerate(destination):
                sendreq = self.comm.Isend(data.flatten(), dest = dest, tag = tag)
                sendreq.wait()

        ## complete the recieve request
        for req in recvreq:
            req.wait()

        return sources, recvbuff