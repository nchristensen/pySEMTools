"""Implements the mesh connectivity class"""

from .msh import Mesh
from ..comm.router import Router
from ..monitoring.logger import Logger
import numpy as np

class MeshConnectivity:
    """
    Class to compute the connectivity of the mesh
    
    Uses facets and vertices to determine which elements are connected to each other

    Parameters
    ----------
    comm : MPI communicator
        The MPI communicator
    msh : Mesh
        The mesh object
    rel_tol : float
        The relative tolerance to use when comparing the coordinates of the facets/edges
    """

    def __init__(self, comm, msh: Mesh = None, rel_tol = 0.01):

        self.log = Logger(comm=comm, module_name="MeshConnectivity")
        self.log.write("info", "Initializing MeshConnectivity")
        self.log.tic()
        self.rt = Router(comm)
        self.rtol = rel_tol

        if isinstance(msh, Mesh):
            # Create local connecitivy
            self.local_connectivity(msh)

            #print(np.unique(self.unique_efp_elem))

            # Create global connectivity
            self.global_connectivity(msh)

        self.log.write("info", "MeshConnectivity initialized")
        self.log.toc()
    
    def local_connectivity(self, msh: Mesh):
        """
        Computes the local connectivity of the mesh
        
        This function checks elements within a rank

        Parameters
        ----------
        msh : Mesh
            The mesh object

        Notes
        -----
        In 3D, the centers of the facets are compared.
        efp means element facet pair.
        One obtains a local_shared_efp_to_elem_map and a local_shared_efp_to_facet_map dictionary.

        local_shared_efp_to_elem_map[(e, f)] = [e1, e2, ...] gives a list with the elements e1, e2 ... 
        that share the same facet f of element e.

        local_shared_efp_to_facet_map[(e, f)] = [f1, f2, ...] gives a list with the facets f1, f2 ... 
        of the elements e1, e2 ... that share the same facet f of element e.

        In each case, the index of the element list, corresponds to the index of the facet list.
        Therefore, the element list might have repeated element entries.

        Additionally, we create a list of unique_efp_elem and unique_efp_facet, which are the elements and facets
        that are not shared with any other element. These are either boundary elements or elements that are connected to other ranks.
        the unique pairs are the ones that are checked in global connecitivy,
        """

        if msh.gdim >= 1:
            
            self.log.write("info", "Computing local connectivity: Using vertices")

            # Allocate local dictionaries
            self.local_shared_evp_to_elem_map = {}
            self.local_shared_evp_to_vertex_map = {}

            # For all elements, check all the vertices
            for e in range(0, msh.nelv):
                    
                ## For each vertex, find any other vertices that match the coordinates
                #for vertex in range(0, msh.vertices.shape[1]):
                #    same_x =  np.isclose(msh.vertices[e, vertex, 0],msh.vertices[:, :, 0], rtol=self.rtol)
                #    same_y =  np.isclose(msh.vertices[e, vertex, 1],msh.vertices[:, :, 1], rtol=self.rtol)
                #    same_z =  np.isclose(msh.vertices[e, vertex, 2],msh.vertices[:, :, 2], rtol=self.rtol)
                #    same_vertex = np.where(same_x & same_y & same_z)
                #    
                #    matching_elem = same_vertex[0]
                #    matching_vertex = same_vertex[1]

                # Compare all vertices of the element at once
                sh = (1, 1, msh.vertices.shape[1])
                sh2 = (-1, msh.vertices.shape[1], 1)
                same_x = np.isclose(msh.vertices[e, :, 0].reshape(sh), msh.vertices[:, :, 0].reshape(sh2), rtol=self.rtol)
                same_y = np.isclose(msh.vertices[e, :, 1].reshape(sh), msh.vertices[:, :, 1].reshape(sh2), rtol=self.rtol)
                same_z = np.isclose(msh.vertices[e, :, 2].reshape(sh), msh.vertices[:, :, 2].reshape(sh2), rtol=self.rtol)                
                same_vertex = np.where(same_x & same_y & same_z)

                # If a match is found, populate the local dictionaries
                if 1==1:
                    my_vertex = same_vertex[2]
                    matching_elem = same_vertex[0]
                    matching_vertex = same_vertex[1]
                    
                    for vertex in range(0, msh.vertices.shape[1]):
                    
                        v_match = np.where(my_vertex == vertex)
                    
                        if len(v_match) > 0:
                            me = matching_elem[v_match]
                            mv = matching_vertex[v_match]
                        else:
                            me = []
                            mv = []

                        self.local_shared_evp_to_elem_map[(e, vertex)] = me
                        self.local_shared_evp_to_vertex_map[(e, vertex)] = mv

            # For all vertices where no match was found, indicate that they are "incomplete"
            # This means they can be boundary or shared with other ranks

            if msh.gdim == 2:
                min_vertex = 4 # Anything less than 4 means that a vertex might be in another rank
            else:
                min_vertex = 8 # Anything less than 8 means that a vertex might be in another rank

            self.incomplete_evp_elem = []
            self.incomplete_evp_vertex = []
            for elem_vertex_pair in self.local_shared_evp_to_elem_map.keys():

                if len(self.local_shared_evp_to_elem_map[elem_vertex_pair]) < min_vertex: 
                    print(elem_vertex_pair, self.local_shared_evp_to_elem_map[elem_vertex_pair], self.local_shared_evp_to_vertex_map[elem_vertex_pair])
                    self.incomplete_evp_elem.append(elem_vertex_pair[0])
                    self.incomplete_evp_vertex.append(elem_vertex_pair[1])
            
            # Delete my own entry from the map.
            # Keep any other shared vertex I have in the map.
            for i in range(0, len(self.incomplete_evp_elem)):
                elem_vertex_pair = (self.incomplete_evp_elem[i], self.incomplete_evp_vertex[i])
                self.local_shared_evp_to_elem_map[elem_vertex_pair] = self.local_shared_evp_to_elem_map[elem_vertex_pair][np.where(self.local_shared_evp_to_elem_map[elem_vertex_pair] != elem_vertex_pair[0])]
                self.local_shared_evp_to_vertex_map[elem_vertex_pair] = self.local_shared_evp_to_vertex_map[elem_vertex_pair][np.where(self.local_shared_evp_to_elem_map[elem_vertex_pair] != elem_vertex_pair[0])]
                    
        if msh.gdim == 3:
            
            self.log.write("info", "Computing local connectivity: Using facet centers")

            # Allocate local dictionaries
            self.local_shared_efp_to_elem_map = {}
            self.local_shared_efp_to_facet_map = {}

            # For all elements, check all the facets
            for e in range(0, msh.nelv):

                # For each facets, find any other facets that match the coordinates
                for facet in range(0, 6):
                    same_x =  np.isclose(msh.facet_centers[e, facet, 0],msh.facet_centers[:, :, 0], rtol=self.rtol)
                    same_y =  np.isclose(msh.facet_centers[e, facet, 1],msh.facet_centers[:, :, 1], rtol=self.rtol)
                    same_z =  np.isclose(msh.facet_centers[e, facet, 2],msh.facet_centers[:, :, 2], rtol=self.rtol)

                    same_facet = np.where(same_x & same_y & same_z)
                    matching_elem = same_facet[0]
                    matching_facet = same_facet[1]

                    self.local_shared_efp_to_elem_map[(e, facet)] = matching_elem
                    self.local_shared_efp_to_facet_map[(e, facet)] = matching_facet

            # For all facets where no match was found, indicate that they are "unique"
            # This means they can be boundary or shared with other ranks
            self.unique_efp_elem = []
            self.unique_efp_facet = []
            for elem_facet_pair in self.local_shared_efp_to_elem_map.keys():

                if len(self.local_shared_efp_to_elem_map[elem_facet_pair]) < 2: # All facets will appear at least once
                    self.unique_efp_elem.append(elem_facet_pair[0])
                    self.unique_efp_facet.append(elem_facet_pair[1])
                    
            # Delete the unique element facet pairs from the local maps
            # Since I just need to sum the values of the matching facet
            for i in range(0, len(self.unique_efp_elem)):
                elem_facet_pair = (self.unique_efp_elem[i], self.unique_efp_facet[i])
                self.local_shared_efp_to_elem_map.pop(elem_facet_pair)
                self.local_shared_efp_to_facet_map.pop(elem_facet_pair)

    def global_connectivity(self, msh: Mesh):
        """ 
        Computes the global connectivity of the mesh
        
        Currently this function sends data from all to all.

        Parameters
        ----------
        msh : Mesh
            The mesh object
        
        Notes
        -----
        In 3D. this function sends the facet centers of the unique_efp_elem and unique_efp_facet to all other ranks.
        as well as the element ID and facet ID to be assigned.

        We compare the unique facet centers of our rank to those of others and determine which one matches.
        When we find that one matches, we populate the directories:
        global_shared_efp_to_rank_map[(e, f)] = rank
        global_shared_efp_to_elem_map[(e, f)] = elem
        global_shared_efp_to_facet_map[(e, f)] = facet

        So for each element facet pair we will know which rank has it, and which is their ID in that rank.

        BE MINDFUL: Later when redistributing, send the points, but also send the element and facet ID to the other ranks so the reciever 
        can know which is the facet that corresponds.
        """
            
        if msh.gdim == 3:

            self.log.write("info", "Computing global connectivity: Using facet centers")
        
            # Send unique facet centers and their element and facet id to all other ranks.
            destinations = [rank for rank in range(0, self.rt.comm.Get_size()) if rank != self.rt.comm.Get_rank()]

            local_unique_el_id = np.array(self.unique_efp_elem)
            local_unique_facet_id = np.array(self.unique_efp_facet)
            local_unique_facet_centers = msh.facet_centers[self.unique_efp_elem, self.unique_efp_facet]

            sources, source_unique_el_id = self.rt.all_to_all(destination=destinations, data=local_unique_el_id, dtype=local_unique_el_id.dtype)
            _ , source_unique_facet_id = self.rt.all_to_all(destination=destinations, data=local_unique_facet_id, dtype=local_unique_facet_id.dtype)
            _, source_unique_facet_centers = self.rt.all_to_all(destination=destinations, data=local_unique_facet_centers, dtype=local_unique_facet_centers.dtype)

            for i in range(0, len(source_unique_facet_centers)):
                source_unique_facet_centers[i] = source_unique_facet_centers[i].reshape(-1, 3)

            # Create global dictionaries
            self.global_shared_efp_to_rank_map = {}
            self.global_shared_efp_to_elem_map = {}
            self.global_shared_efp_to_facet_map = {}

            # Go through the data in each other rank.
            for source_idx, source_fc in enumerate(source_unique_facet_centers):

                remove_pair_idx = []

                # Loop through all my own unique element facet pair coordinates
                for e_f_pair in range(0, len(self.unique_efp_elem)):

                    # Check where my unique facet pair coordinates match with the unique ...
                    # ... facet pair coordinates of the other rank
                    e = self.unique_efp_elem[e_f_pair]
                    facet = self.unique_efp_facet[e_f_pair]
                    same_x =  np.isclose(msh.facet_centers[e, facet, 0],source_fc[:, 0], rtol=self.rtol)
                    same_y =  np.isclose(msh.facet_centers[e, facet, 1],source_fc[:, 1], rtol=self.rtol)
                    same_z =  np.isclose(msh.facet_centers[e, facet, 2],source_fc[:, 2], rtol=self.rtol)
                    same_facet = np.where(same_x & same_y & same_z)
                        
                    # If we find a match assign it in the global dictionaries
                    if len(same_facet[0]) > 0:
                        matching_id = same_facet[0]
                        self.global_shared_efp_to_rank_map[(e, facet)] = sources[source_idx]
                        self.global_shared_efp_to_elem_map[(e, facet)] = source_unique_el_id[source_idx][matching_id]
                        self.global_shared_efp_to_facet_map[(e, facet)] = source_unique_facet_id[source_idx][matching_id]
                        remove_pair_idx.append(e_f_pair)

                # If a match is found for an element facet pair, remove it from the list of unique pairs
                remove_pair_idx = sorted(remove_pair_idx, reverse=True)
                for idx in remove_pair_idx:
                    self.unique_efp_elem.pop(idx)
                    self.unique_efp_facet.pop(idx)
                
            # Any of my unique facets that remain, should be a boundary facet
            self.boundary_efp_elem = self.unique_efp_elem
            self.boundary_efp_facet = self.unique_efp_facet
            del self.unique_efp_elem
            del self.unique_efp_facet


