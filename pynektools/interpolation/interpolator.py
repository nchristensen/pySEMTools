import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from tqdm import tqdm
from .mpi_ops import *
from .sem import element_interpolator_c
from mpi4py import MPI # for the timer
from itertools import combinations

NoneType = type(None)

# Variables for debugging
debug = False
tr = 7

class interpolator_c():
    def __init__(self, x, y, z, probes, comm, progress_bar = False, modal_search = True):

        self.x = x
        self.y = y
        self.z = z
        self.probes = probes
        self.ei = element_interpolator_c(x.shape[1], modal_search = modal_search)
        self.progress_bar = progress_bar
        
        #Find the element offset of each rank so you can store the global element number
        nelv = self.x.shape[0]
        sendbuf = np.ones((1), np.intc) * nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - nelv


    def scatter_probes_from_io_rank(self, io_rank, comm):
        
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Check how many probes should be in each rank with a load balanced linear distribution 
        probe_partition_sendcount = np.zeros((size), dtype = np.intc)
        if rank == io_rank:
            for i_rank in range(0, size):
                M = self.probes.shape[0]
                pe_rank = i_rank
                pe_size = comm.Get_size()
                L = np.floor(np.double(M) / np.double(pe_size))
                R = np.mod(M, pe_size)
                Ip = np.floor((np.double(M) + np.double(pe_size) - np.double(pe_rank) - np.double(1)) / np.double(pe_size))
                nelv = int(Ip)
                offset_el = int(pe_rank * L + min(pe_rank, R))
                n = 3 * nelv
                probe_partition_sendcount[i_rank] = int(nelv)
        
        comm.Bcast(probe_partition_sendcount, root=io_rank)
        probe_coord_partition_sendcount = probe_partition_sendcount * 3 # Since each probe has 3 coordinates


        # Define some variables that need to be only in the rank that partitions
        if rank == io_rank:
            # Set the necesary arrays for identification of point
            number_of_points = self.probes.shape[0]
            self.probes_rst = np.zeros((number_of_points, 3), dtype = np.double)
            self.el_owner = np.zeros((number_of_points), dtype = np.intc)
            self.glb_el_owner = np.zeros((number_of_points), dtype = np.intc)
            #self.rank_owner = np.zeros((number_of_points), dtype = np.intc)
            self.rank_owner = np.ones((number_of_points), dtype = np.intc)*-1000
            self.err_code = np.zeros((number_of_points), dtype = np.intc) # 0 not found, 1 is found
            self.test_pattern = np.ones((number_of_points), dtype = np.double) # test interpolation holder
        else:
            self.probes_rst = None
            self.el_owner = None
            self.glb_el_owner = None
            self.rank_owner = None
            self.err_code = None
            self.test_pattern = None

        # Now scatter the probes and all the codes according to what is needed and each rank has a partition
        self.probe_partition_sendcount = probe_partition_sendcount
        self.probe_coord_partition_sendcount = probe_coord_partition_sendcount
        ## Double
        tmp = scatter_from_root(self.probes, probe_coord_partition_sendcount, io_rank, np.double, comm)
        self.probe_partition = tmp.reshape((int(tmp.size/3),3))
        tmp = scatter_from_root(self.probes_rst, probe_coord_partition_sendcount, io_rank, np.double, comm)
        self.probe_rst_partition = tmp.reshape((int(tmp.size/3),3))
        ## Int
        self.el_owner_partition = scatter_from_root(self.el_owner, probe_partition_sendcount, io_rank, np.intc, comm)
        self.glb_el_owner_partition = scatter_from_root(self.glb_el_owner, probe_partition_sendcount, io_rank, np.intc, comm)
        self.rank_owner_partition = scatter_from_root(self.rank_owner, probe_partition_sendcount, io_rank, np.intc, comm)
        self.err_code_partition = scatter_from_root(self.err_code, probe_partition_sendcount, io_rank, np.intc, comm)
        ## Double
        self.test_pattern_partition = scatter_from_root(self.test_pattern, probe_partition_sendcount, io_rank, np.double, comm)
        
        return
    
    def gather_probes_to_io_rank(self, io_rank, comm):
        
        rank = comm.Get_rank()
        size = comm.Get_size()

        root = io_rank
        sendbuf = self.probe_partition.reshape((self.probe_partition.size))
        recvbuf, _ = gather_in_root(sendbuf, root, np.double,  comm)
        if type(recvbuf) != NoneType:
            self.probes[:,:] = recvbuf.reshape((int(recvbuf.size/3),3))[:,:]
        sendbuf = self.probe_rst_partition.reshape((self.probe_rst_partition.size))
        recvbuf, _ = gather_in_root(sendbuf, root, np.double,  comm)
        if type(recvbuf) != NoneType:
            self.probes_rst[:,:] = recvbuf.reshape((int(recvbuf.size/3),3))[:,:]
        sendbuf = self.el_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc,  comm)
        if type(recvbuf) != NoneType:
            self.el_owner[:] = recvbuf[:]
        sendbuf = self.glb_el_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc,  comm)
        if type(recvbuf) != NoneType:
            self.glb_el_owner[:] = recvbuf[:]
        sendbuf = self.rank_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc,  comm)
        if type(recvbuf) != NoneType:
            self.rank_owner[:] = recvbuf[:]
        sendbuf = self.err_code_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc,  comm)
        if type(recvbuf) != NoneType:
            self.err_code[:] = recvbuf[:]
        sendbuf = self.test_pattern_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.double,  comm)
        if type(recvbuf) != NoneType:
            self.test_pattern[:] = recvbuf[:]
        
        return


    def find_points(self, comm, use_kdtree = True, test_tol = 1e-4):
        
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.rank = rank
        
        # First each rank finds their bounding box 
        self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        if use_kdtree: 
            # Get bbox centroids and max radius from center to corner
            self.my_bbox_centroids, self.my_bbox_maxdist = get_bbox_centroids_and_max_dist(self.my_bbox)

            # Build a KDtree with my information
            self.my_tree = KDTree(self.my_bbox_centroids)

        nelv = self.x.shape[0]
        self.ranks_ive_checked = []

        # Create a directory that contains which are the colours for each rank in each iteration
        rank_dict, colour_dict = get_communication_pairs(comm)

        # Get candidate ranks from a global kd tree
        candidate_ranks =  get_candidate_ranks(self, comm)

        ## Print for debugging
        #if rank == 0 :
        #    for rankk in rank_dict:
        #        print("Rank: ", rankk)
        #        print(rank_dict[rankk])
        #    print("=======")
        #    
        #    for rankk in colour_dict:
        #        print("Rank: ", rankk)
        #        print(colour_dict[rankk])


        # Iterate over the pairs. Currently it is assumed that the size is a power of two, therefore all ranks need the same iterations
        number_of_its = len(colour_dict[rank])
        j = 0
        while j < number_of_its :

            # Get the colour from the dictionary
            col = colour_dict[rank][j]

            if debug:
                print("rank: {}, finding points. start iteration: {}. Color: {}".format(rank, j, col))
            else:
                if rank == 0: print("rank: {}, finding points. start iteration: {}. Color: {}".format(rank, j, col))
            start_time = MPI.Wtime()
            
            # Split the communicator
            search_comm= comm.Split(color = col, key=rank)
            search_rank = search_comm.Get_rank()
            search_size = search_comm.Get_size()

            # Get the send_recv ranks in this iteration
            sr = [rank_dict[rank][j][0], rank_dict[rank][j][1]]

            # See which one is the other rank in the communicator
            if j == 0:
                # In the first iteration, I should check my own rank, however
                other_rank = [x for x in sr]
            else:
                other_rank = [x for x in sr if x!= rank]

            # Check if this other rank is in my candidates
            communicate_data = np.zeros((1), dtype = np.intc)
            for r in other_rank:
                if r in candidate_ranks:
                    communicate_data[0] = 1

            # Retrieve the information from the other rank in the pair
            comm_communicate_data = np.zeros((search_size), dtype=np.intc)
            search_comm.Allgather([communicate_data, MPI.INT], [comm_communicate_data, MPI.INT])

            # Now, if one of the ranks is candidate for the other, then keep going with the iterations, otherwise do not
            communicate = 0
            comm_list = comm_communicate_data.tolist()
            for c in comm_list:
                if c == 1:
                    communicate = 1

            if communicate != 1:
                j = j + 1
                continue

            # Make each rank in the communicator broadcast their bounding boxes to the others to search
            for broadcaster in range(0, search_size):

                # Points that I have not found to which rank or element belong to
                not_found = np.where(self.err_code_partition != 1)[0]
                n_not_found = not_found.size
                probe_not_found = self.probe_partition[not_found] 
                probe_rst_not_found = self.probe_rst_partition[not_found] 
                el_owner_not_found = self.el_owner_partition[not_found] 
                glb_el_owner_not_found = self.glb_el_owner_partition[not_found] 
                rank_owner_not_found = self.rank_owner_partition[not_found] 
                err_code_not_found = self.err_code_partition[not_found] 
                test_pattern_not_found = self.test_pattern_partition[not_found] 
        
                # Tell every rank in the broadcaster of the local communicator their actual rank
                broadcaster_global_rank = np.ones((1), dtype=np.intc)*rank
                search_comm.Bcast(broadcaster_global_rank, root=broadcaster)

                # Tell every rank how much they need to allocate for the broadcaster bounding boxes
                nelv_in_broadcaster = np.ones((1), dtype=np.intc)*nelv
                search_comm.Bcast(nelv_in_broadcaster, root=broadcaster)

                # Allocate the recieve buffer for bounding boxes
                bbox_rec_buff = np.empty((nelv_in_broadcaster[0], 6), dtype = np.double)

                # Only in the broadcaster copy the data
                if search_rank == broadcaster: bbox_rec_buff[:,:] = np.copy(self.my_bbox[:,:])

                # Broadcast the bounding boxes so each rank can check if the points are there
                search_comm.Bcast(bbox_rec_buff, root=broadcaster)
     
                # Only do the search, if my rank has not already searched the broadcaster
                if broadcaster_global_rank not in self.ranks_ive_checked: 

                    #================================================================================
                    if not use_kdtree:
                        # Find a candidate rank to check
                        i = 0
                        if self.progress_bar: pbar= tqdm(total=n_not_found)
                        for pt in probe_not_found:
                            found_candidate = False
                            for e in range(0, bbox_rec_buff.shape[0]):
                                if pt_in_bbox(pt, bbox_rec_buff[e]):
                                    found_candidate = True
                                    #el_owner_not_found[i] = e
                                    err_code_not_found[i] = 1
                                    #rank_owner_not_found[i] = broadcaster_global_rank

                            if found_candidate == True:
                                i = i + 1
                                if self.progress_bar: pbar.update(1)
                                continue
                            i = i + 1
                            if self.progress_bar: pbar.update(1)
                        if self.progress_bar: pbar.close()
                    #================================================================================
                    elif use_kdtree:

                        # Get bbox centroids and max radius from center to corner for the broadcaster
                        bbox_centroids, bbox_maxdist = get_bbox_centroids_and_max_dist(bbox_rec_buff)
                
                        # Create the KDtree
                        broadcaster_tree = KDTree(bbox_centroids)
                        # Query the tree with the probes to reduce the bbox search
                        candidate_elements = broadcaster_tree.query_ball_point(x=probe_not_found, r=bbox_maxdist, p=2.0, eps=1e-8, workers=1, return_sorted=False, return_length=False)

                        # Do a bbox search over the candidate elements, just as it used to be done (The KD tree allows to avoid searching ALL elements)
                        i = 0
                        if self.progress_bar: pbar= tqdm(total=n_not_found)
                        for pt in probe_not_found:
                            found_candidate = False
                            for e in candidate_elements[i]:
                                if pt_in_bbox(pt, bbox_rec_buff[e]):
                                    found_candidate = True
                                    #el_owner_not_found[i] = e
                                    err_code_not_found[i] = 1
                                    #rank_owner_not_found[i] = broadcaster_global_rank

                            if found_candidate == True:
                                i = i + 1
                                if self.progress_bar: pbar.update(1)
                                continue
                            i = i + 1
                            if self.progress_bar: pbar.update(1)
                        if self.progress_bar: pbar.close()
                    #================================================================================


                    #Now let the brodcaster gather the points that the other ranks think it has
                    #broadcaster_is_candidate = np.where(rank_owner_not_found == broadcaster_global_rank)[0] 
                    broadcaster_is_candidate = np.where(err_code_not_found == 1)[0] 
            
                    self.ranks_ive_checked.append(broadcaster_global_rank[0])
                else:
                    #If this rank has already checked the broadcaster, just produce an empty list
                    #broadcaster_is_candidate = np.where(rank_owner_not_found == -10000)[0] 
                    broadcaster_is_candidate = np.where(err_code_not_found == 10000)[0] 

                probe_broadcaster_is_candidate = probe_not_found[broadcaster_is_candidate]
                probe_rst_broadcaster_is_candidate = probe_rst_not_found[broadcaster_is_candidate]
                el_owner_broadcaster_is_candidate = el_owner_not_found[broadcaster_is_candidate]
                glb_el_owner_broadcaster_is_candidate = glb_el_owner_not_found[broadcaster_is_candidate]
                rank_owner_broadcaster_is_candidate =   rank_owner_not_found[broadcaster_is_candidate]
                err_code_broadcaster_is_candidate = err_code_not_found[broadcaster_is_candidate] 
                test_pattern_broadcaster_is_candidate = test_pattern_not_found[broadcaster_is_candidate] 
                 
                root = broadcaster
                tmp, probe_sendcount_broadcaster_is_candidate = gather_in_root(probe_broadcaster_is_candidate.reshape((probe_broadcaster_is_candidate.size)), root, np.double,  search_comm)
                if type(tmp) != NoneType:
                    probe_broadcaster_has = tmp.reshape((int(tmp.size/3),3))
                
                # For debugging
                if broadcaster_global_rank == tr and rank == tr and debug == True: 
                    print(probe_sendcount_broadcaster_is_candidate/3)
                    #print(probe_broadcaster_has)
                
                tmp, _ = gather_in_root(probe_rst_broadcaster_is_candidate.reshape((probe_rst_broadcaster_is_candidate.size)), root, np.double,  search_comm)
                if type(tmp) != NoneType:
                    probe_rst_broadcaster_has = tmp.reshape((int(tmp.size/3),3)) 
                el_owner_broadcaster_has, el_owner_sendcount_broadcaster_is_candidate = gather_in_root(el_owner_broadcaster_is_candidate, root, np.intc,  search_comm)
                glb_el_owner_broadcaster_has, _ = gather_in_root(glb_el_owner_broadcaster_is_candidate, root, np.intc,  search_comm)
                rank_owner_broadcaster_has, _ = gather_in_root(rank_owner_broadcaster_is_candidate, root, np.intc,  search_comm)
                err_code_broadcaster_has, _ = gather_in_root(err_code_broadcaster_is_candidate, root, np.intc,  search_comm)
                test_pattern_broadcaster_has, _ = gather_in_root(test_pattern_broadcaster_is_candidate, root, np.double,  search_comm)
                

                # Now let the broadcaster check if it really had the point. It will go through all the elements again and check rst coordinates
                if search_rank == broadcaster:
                    probe_broadcaster_has, probe_rst_broadcaster_has, el_owner_broadcaster_has, glb_el_owner_broadcaster_has, rank_owner_broadcaster_has, err_code_broadcaster_has, test_pattern_broadcaster_has = self.find_rst(probe_broadcaster_has, probe_rst_broadcaster_has, el_owner_broadcaster_has, glb_el_owner_broadcaster_has, rank_owner_broadcaster_has, err_code_broadcaster_has, test_pattern_broadcaster_has, broadcaster_global_rank, self.offset_el,  not_found_code = -10, use_kdtree = use_kdtree)

                # Now scatter the results back to all the other ranks 
                root = broadcaster
                if search_rank == root:
                    sendbuf = probe_broadcaster_has.reshape(probe_broadcaster_has.size)
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(sendbuf, probe_sendcount_broadcaster_is_candidate, root, np.double, search_comm) 
                probe_broadcaster_is_candidate[:,:] = recvbuf.reshape((int(recvbuf.size/3), 3))[:,:]
                if search_rank == root:
                    sendbuf = probe_rst_broadcaster_has.reshape(probe_rst_broadcaster_has.size)
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(sendbuf, probe_sendcount_broadcaster_is_candidate, root, np.double, search_comm) 
                probe_rst_broadcaster_is_candidate[:,:] = recvbuf.reshape((int(recvbuf.size/3), 3))[:,:]
                if search_rank == root:
                    sendbuf = el_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(sendbuf, el_owner_sendcount_broadcaster_is_candidate, root, np.intc, search_comm) 
                el_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    #sendbuf = el_owner_broadcaster_has+self.offset_el 
                    sendbuf = glb_el_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(sendbuf, el_owner_sendcount_broadcaster_is_candidate, root, np.intc, search_comm) 
                glb_el_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = rank_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(sendbuf, el_owner_sendcount_broadcaster_is_candidate, root, np.intc, search_comm) 
                rank_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = err_code_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(sendbuf, el_owner_sendcount_broadcaster_is_candidate, root, np.intc, search_comm) 
                err_code_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = test_pattern_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(sendbuf, el_owner_sendcount_broadcaster_is_candidate, root, np.double, search_comm) 
                test_pattern_broadcaster_is_candidate[:] = recvbuf[:]
                                
                # Now that the data is back at the original ranks, put it in the place of the not found list that it should be
                probe_not_found[broadcaster_is_candidate,:] = probe_broadcaster_is_candidate[:,:]                 
                probe_rst_not_found[broadcaster_is_candidate,:] = probe_rst_broadcaster_is_candidate[:,:]         
                el_owner_not_found[broadcaster_is_candidate] = el_owner_broadcaster_is_candidate[:]           
                glb_el_owner_not_found[broadcaster_is_candidate] = glb_el_owner_broadcaster_is_candidate[:]                
                rank_owner_not_found[broadcaster_is_candidate] = rank_owner_broadcaster_is_candidate[:]                 
                err_code_not_found[broadcaster_is_candidate] = err_code_broadcaster_is_candidate[:]  
                test_pattern_not_found[broadcaster_is_candidate] = test_pattern_broadcaster_is_candidate[:]  

                # at the end of the broadcaster run, update the information from the previously not found data
                self.probe_partition[not_found,:] = probe_not_found[:,:] 
                self.probe_rst_partition[not_found] = probe_rst_not_found[:,:]
                self.el_owner_partition[not_found] = el_owner_not_found[:]
                self.glb_el_owner_partition[not_found] = glb_el_owner_not_found[:]
                self.rank_owner_partition[not_found] = rank_owner_not_found[:]
                self.err_code_partition[not_found] = err_code_not_found[:]
                self.test_pattern_partition[not_found] = test_pattern_not_found[:]
           
            if debug:
                print("rank: {}, finding points. finished iteration: {}. time(s): {}".format(rank, j, MPI.Wtime()-start_time))
            else:
                if rank == 0: print("rank: {}, finding points. finished iteration: {}. time(s): {}".format(rank, j, MPI.Wtime()-start_time))

            j = j + 1
        

        # Final check
        for j in range(0, len(self.test_pattern_partition)):
            #After all iteration are done, check if some points were not found. Use the error code and the test pattern
            if (self.err_code_partition[j] != 1 and self.test_pattern_partition[j] > test_tol):
                self.err_code_partition[j] = 0
        
            #Check also if the rst are too big, then it needs to be outside
            #if ( abs(self.probe_rst_partition[j, 0]) +  abs(self.probe_rst_partition[j, 1]) +  abs(self.probe_rst_partition[j, 2]) ) > 3.5:
            #    self.err_code_partition[j] = 0



        return
    
    def find_rst(self, probes, probes_rst, el_owner, glb_el_owner, rank_owner, err_code, test_pattern, rank, offset_el, not_found_code = -10, use_kdtree = True, test_interp = True):
        
        # Reset the element owner and the error code so this rank checks again
        #el_owner[:] = 0
        err_code[:] = not_found_code

        # Find the elements in this rank that could hold the point. Check all elements even if you had a guide as an input
        #================================================================================
        if not use_kdtree:
            element_candidates = []
            i = 0
            if self.progress_bar: pbar= tqdm(total=probes.shape[0])
            for pt in probes:
                element_candidates.append([])        
                for e in range(0, self.my_bbox.shape[0]):
                    if pt_in_bbox(pt, self.my_bbox[e]):
                        element_candidates[i].append(e)
                i = i + 1
                if self.progress_bar: pbar.update(1)
            if self.progress_bar: pbar.close()
        #================================================================================
        elif use_kdtree:
                
            # Query the tree with the probes to reduce the bbox search
            candidate_elements = self.my_tree.query_ball_point(x=probes, r=self.my_bbox_maxdist, p=2.0, eps=1e-8, workers=1, return_sorted=False, return_length=False)
            
            element_candidates = []
            i = 0
            if self.progress_bar: pbar= tqdm(total=probes.shape[0])
            for pt in probes:
                element_candidates.append([])        
                for e in candidate_elements[i]:
                    if pt_in_bbox(pt, self.my_bbox[e]):
                        element_candidates[i].append(e)
                i = i + 1
                if self.progress_bar: pbar.update(1)
            if self.progress_bar: pbar.close() 
        #================================================================================
        #if self.rank == tr and debug == True: print(element_candidates)

        if self.progress_bar: pbar= tqdm(total=probes.shape[0])
        for pts in range(0, probes.shape[0]):
            if err_code[pts] != 1:
                for e in element_candidates[pts]:
                    self.ei.project_element_into_basis(self.x[e,:,:,:], self.y[e,:,:,:], self.z[e,:,:,:])
                    r, s, t = self.ei.find_rst_from_xyz(probes[pts,0], probes[pts,1], probes[pts,2]) 
                    if self.ei.point_inside_element:
                        probes_rst[pts, 0] = r
                        probes_rst[pts, 1] = s
                        probes_rst[pts, 2] = t
                        el_owner[pts] = e
                        glb_el_owner[pts] = e + offset_el
                        rank_owner[pts] = rank
                        err_code[pts] = 1                    
                        break
                    else:

                        # Perform test interpolation and update if the results are better than previously stored
                        if test_interp:
                            test_field = self.x[e,:,:,:]**2 + self.y[e,:,:,:]**2 + self.z[e,:,:,:]**2
                            test_probe = probes[pts, 0]**2 +  probes[pts, 1]**2 + probes[pts, 2]**2 
                            test_interp = self.ei.interpolate_field_at_rst(r, s, t, test_field)

                            test_error = abs(test_probe - test_interp)

                            if test_error < test_pattern[pts]:
                                probes_rst[pts, 0] = r
                                probes_rst[pts, 1] = s
                                probes_rst[pts, 2] = t
                                el_owner[pts] = e
                                glb_el_owner[pts] = e + offset_el
                                rank_owner[pts] = rank
                                err_code[pts] = not_found_code
                                test_pattern[pts] = test_error

                        # Otherwise progressively update
                        else:
                            probes_rst[pts, 0] = r
                            probes_rst[pts, 1] = s
                            probes_rst[pts, 2] = t
                            el_owner[pts] = e
                            glb_el_owner[pts] = e + offset_el
                            rank_owner[pts] = rank
                            err_code[pts] = not_found_code


            if self.progress_bar: pbar.update(1)
        if self.progress_bar: pbar.close()
        
        return probes, probes_rst, el_owner, glb_el_owner, rank_owner, err_code, test_pattern



    def redistribute_probes_to_owners_from_io_rank(self, io_rank, comm):

        rank = comm.Get_rank()
        size = comm.Get_size()
        
        probes = self.probes
        probes_rst = self.probes_rst
        el_owner = self.el_owner
        rank_owner = self.rank_owner
        err_code = self.err_code

        # Sort the points by rank to scatter them easily
        if rank == io_rank:

            # Before the sorting, assign all the not found probes to rank zero.
            # The points with error code 0 will be ignored in the interpolation routine
            # Doing this will avoid an error when after interpolating we try to gather the points
            self.rank_owner[np.where(err_code == 0)] = 0
            rank_owner[np.where(err_code == 0)] = 0

            sort_by_rank = np.argsort(rank_owner)

            sorted_probes = probes[sort_by_rank]
            sorted_probes_rst = probes_rst[sort_by_rank]
            sorted_el_owner = el_owner[sort_by_rank]
            sorted_rank_owner = rank_owner[sort_by_rank]
            sorted_err_code = err_code[sort_by_rank]
        else:
            sort_by_rank = None

        # Check the sendcounts in number of probes
        sendcounts = np.zeros((size), dtype = np.intc)
        if rank == io_rank:
            for root in range(0, size):
                sendcounts[root] = len(np.where(rank_owner == root)[0])    
        comm.Bcast(sendcounts, root=0)

        root = io_rank
        # Redistribute probes
        if rank == root:
            sendbuf = sorted_probes.reshape((sorted_probes.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts*3, root, np.double, comm)
        my_probes = recvbuf.reshape((int(recvbuf.size/3), 3))
        
        # Redistribute probes rst
        if rank == root:
            sendbuf = sorted_probes_rst.reshape((sorted_probes_rst.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts*3, root, np.double, comm)
        my_probes_rst = recvbuf.reshape((int(recvbuf.size/3), 3))
    
        # Redistribute err_code
        if rank == root:
            sendbuf = sorted_err_code.reshape((sorted_err_code.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        my_err_code = recvbuf
    
        # Redistribute el_owner
        if rank == root:
            sendbuf = sorted_el_owner.reshape((sorted_el_owner.size))
            #print(sendbuf)
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        #print(recvbuf)
        my_el_owner = recvbuf
    
        # Redistribute el_owner
        if rank == root:
            sendbuf = sorted_rank_owner.reshape((sorted_rank_owner.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        my_rank_owner = recvbuf
    
        self.my_probes = my_probes
        self.my_probes_rst = my_probes_rst
        self.my_err_code = my_err_code
        self.my_el_owner = my_el_owner
        self.my_rank_owner = my_rank_owner
        self.sendcounts = sendcounts
        self.sort_by_rank = sort_by_rank

        return

    
    def interpolate_field_from_rst(self, sampled_field):

        x = self.x
        y = self.y
        z = self.z
        my_probes = self.my_probes
        my_probes_rst = self.my_probes_rst
        my_err_code = self.my_err_code
        sampled_field_at_probe = np.empty((my_probes.shape[0]))
        
        i = 0 #Counter for the number of probes
        if self.progress_bar: pbar= tqdm(total=my_probes.shape[0])
        for e in self.my_el_owner:
            if my_err_code[i] != 0 :
                #self.ei.project_element_into_basis(x[e,:,:,:], y[e,:,:,:], z[e,:,:,:])
                tmp = self.ei.interpolate_field_at_rst(my_probes_rst[i,0], my_probes_rst[i,1], my_probes_rst[i,2], sampled_field[e,:,:,:])
                sampled_field_at_probe[i] = tmp
            else:
                sampled_field_at_probe[i] = 0
            i = i+1
            if self.progress_bar: pbar.update(1)
        if self.progress_bar: pbar.close()
 
        return sampled_field_at_probe


def pt_in_bbox(pt, bbox):
                            
    state = False
    found_x = False
    found_y = False
    found_z = False
                     
    if pt[0] >= bbox[0] and pt[0] <= bbox[1]: 
        found_x=True
    if pt[1] >= bbox[2] and pt[1] <= bbox[3]: 
        found_y=True
    if pt[2] >= bbox[4] and pt[2] <= bbox[5]: 
        found_z=True

    if found_x == True and found_y == True and found_z == True: 
        state = True
    else:
        state = False

    return state

def get_bbox_from_coordinates(x, y, z):

    nelv = x.shape[0]
    lx = x.shape[1]
    ly = x.shape[2]
    lz = x.shape[3]

    bbox = np.zeros((nelv, 6), dtype = np.double)

    for e in range(0, nelv): 
        bbox[e, 0] = np.min(x[e,:,:,:])
        bbox[e, 1] = np.max(x[e,:,:,:])
        bbox[e, 2] = np.min(y[e,:,:,:])
        bbox[e, 3] = np.max(y[e,:,:,:])
        bbox[e, 4] = np.min(z[e,:,:,:])
        bbox[e, 5] = np.max(z[e,:,:,:])

    return bbox


def get_bbox_centroids_and_max_dist(bbox):

    # Then find the centroids of each bbox and the maximun bbox radious from centroid to corner
    bbox_dist = np.zeros((bbox.shape[0], 3))
    bbox_dist[:,0] =   (bbox[:,1] - bbox[:,0]) 
    bbox_dist[:,1] =   (bbox[:,3] - bbox[:,2]) 
    bbox_dist[:,2] =   (bbox[:,5] - bbox[:,4])

    bbox_max_dist = np.max(np.sqrt(bbox_dist[:,0]**2 + bbox_dist[:,1]**2 + bbox_dist[:,2]**2)/2)

    bbox_centroid = np.zeros((bbox.shape[0], 3))
    bbox_centroid[:,0] =   bbox[:,0] + bbox_dist[:,0]/2 
    bbox_centroid[:,1] =   bbox[:,2] + bbox_dist[:,1]/2 
    bbox_centroid[:,2] =   bbox[:,4] + bbox_dist[:,2]/2 

    return bbox_centroid, bbox_max_dist


def get_communication_pairs(comm):

    size = comm.Get_size()

    # Create a list with all the ranks
    all_ranks = [ii for ii in range(0, size)]

    # Create a dictionary to hold the pairs and colors for each rank
    rank_dict = {i: [(i,i)] for i in range(size)}
    colour_dict = {i: [i] for i in range(size)}

    # Get all unique pairs and colours
    all_pairs = list(combinations(all_ranks, 2))
    all_colours = [xx for xx in range(0, len(all_pairs))]

    # Iterate over all the unique pairs to see which ranks communicate each iteration
    while len(all_pairs) > 0:
        rank_send_recv = [] 
        included_pairs = []
        included_col_pairs = []
        for ii in range (0, len(all_pairs)): 
            rs = all_pairs[ii][0]
            rr = all_pairs[ii][1]
            if rs in rank_send_recv or rr in rank_send_recv:
                a = 1
            else: 
                # Update the pair and colour for this rank
                rank_dict[rs].append(all_pairs[ii])
                rank_dict[rr].append(all_pairs[ii])
                colour_dict[rs].append(all_colours[ii])
                colour_dict[rr].append(all_colours[ii])

                # Keep track of the ranks that have been included
                included_pairs.append(all_pairs[ii])
                included_col_pairs.append(all_colours[ii])
                    
                # Update the list of ranks recieved and sent this iteration
                rank_send_recv.append(rs)
                rank_send_recv.append(rr)
            
        # Now remove the pairs that have been included this iteration
        for pair in included_pairs:
            all_pairs.remove(pair) 
        for pair in included_col_pairs:
            all_colours.remove(pair)

    return rank_dict, colour_dict
        
def get_candidate_ranks(self, comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Find the bounding box of the rank to create a global but "sparse" kdtree       
    rank_bbox = np.zeros((1, 6), dtype=np.double)
    rank_bbox[0,0] = np.min(self.x)
    rank_bbox[0,1] = np.max(self.x)
    rank_bbox[0,2] = np.min(self.y)
    rank_bbox[0,3] = np.max(self.y)
    rank_bbox[0,4] = np.min(self.z)
    rank_bbox[0,5] = np.max(self.z)
        
    rank_bbox_dist = np.zeros((1, 3), dtype=np.double)
    rank_bbox_dist[0,0] =   (rank_bbox[0,1] - rank_bbox[0,0])
    rank_bbox_dist[0,1] =   (rank_bbox[0,3] - rank_bbox[0,2])
    rank_bbox_dist[0,2] =   (rank_bbox[0,5] - rank_bbox[0,4])
    rank_bbox_max_dist = np.max(np.sqrt(rank_bbox_dist[:,0]**2 + rank_bbox_dist[:,1]**2 + rank_bbox_dist[:,2]**2)/2)

    rank_bbox_centroid = np.zeros((1, 3))
    rank_bbox_centroid[:,0] =   rank_bbox[:,0] + rank_bbox_dist[:,0]/2 
    rank_bbox_centroid[:,1] =   rank_bbox[:,2] + rank_bbox_dist[:,1]/2 
    rank_bbox_centroid[:,2] =   rank_bbox[:,4] + rank_bbox_dist[:,2]/2 
 
    global_centroids = np.zeros((size*3), dtype=np.double)
    global_max_dist = np.zeros((size), dtype=np.double)

    # Gather in all ranks
    comm.Allgather([rank_bbox_centroid.flatten(), MPI.DOUBLE], [global_centroids, MPI.DOUBLE])
    comm.Allgather([rank_bbox_max_dist, MPI.DOUBLE], [global_max_dist, MPI.DOUBLE])
    global_centroids = global_centroids.reshape((size, 3))
    # Create a tree with the rank centroids
    self.global_tree = KDTree(global_centroids) 
    candidate_ranks_per_point = self.global_tree.query_ball_point(x=self.probe_partition, r=np.max(global_max_dist), p=2.0, eps=1e-8, workers=1, return_sorted=False, return_length=False)

    flattened_list = [item for sublist in candidate_ranks_per_point for item in sublist]
    candidate_ranks = list(set(flattened_list))

    return candidate_ranks
