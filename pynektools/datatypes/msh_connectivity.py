"""Implements the mesh connectivity class"""

from .msh import Mesh
from .coef import Coef
from ..comm.router import Router
from ..monitoring.logger import Logger
import numpy as np
from .element_slicing import fetch_elem_facet_data as fd
from .element_slicing import fetch_elem_edge_data as ed
from .element_slicing import fetch_elem_vertex_data as vd
from .element_slicing import vertex_to_slice_map_2d, vertex_to_slice_map_3d, edge_to_slice_map_2d, edge_to_slice_map_3d, facet_to_slice_map
import sys

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

    def __init__(self, comm, msh: Mesh = None, rel_tol = 1e-5):

        self.log = Logger(comm=comm, module_name="MeshConnectivity")
        self.log.write("info", "Initializing MeshConnectivity")
        self.log.tic()
        self.rt = Router(comm)
        self.rtol = rel_tol

        if isinstance(msh, Mesh):
            # Create local connecitivy
            self.local_connectivity(msh)

            # Create global connectivity
            self.global_connectivity(msh)

            # Get the multiplicity
            self.get_multiplicity(msh)

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
                #''' 
                # For each vertex, find any other vertices that match the coordinates
                for vertex in range(0, msh.vertices.shape[1]):
                    same_x =  np.isclose(msh.vertices[e, vertex, 0],msh.vertices[:, :, 0], rtol=self.rtol)
                    same_y =  np.isclose(msh.vertices[e, vertex, 1],msh.vertices[:, :, 1], rtol=self.rtol)
                    same_z =  np.isclose(msh.vertices[e, vertex, 2],msh.vertices[:, :, 2], rtol=self.rtol)
                    same_vertex = np.where(same_x & same_y & same_z)
                    
                    matching_elem = same_vertex[0]
                    matching_vertex = same_vertex[1]
                    
                    self.local_shared_evp_to_elem_map[(e, vertex)] = matching_elem
                    self.local_shared_evp_to_vertex_map[(e, vertex)] = matching_vertex

                '''
                # Compare all vertices of the element at once
                sh = (1, 1, msh.vertices.shape[1])
                sh2 = (-1, msh.vertices.shape[1], 1)
                same_x = np.isclose(msh.vertices[e, :, 0].reshape(sh), msh.vertices[:, :, 0].reshape(sh2), rtol=self.rtol)
                same_y = np.isclose(msh.vertices[e, :, 1].reshape(sh), msh.vertices[:, :, 1].reshape(sh2), rtol=self.rtol)
                same_z = np.isclose(msh.vertices[e, :, 2].reshape(sh), msh.vertices[:, :, 2].reshape(sh2), rtol=self.rtol)                
                same_vertex = np.where(same_x & same_y & same_z)

                # If a match is found, populate the local dictionaries
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
                '''
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
                    self.incomplete_evp_elem.append(elem_vertex_pair[0])
                    self.incomplete_evp_vertex.append(elem_vertex_pair[1])
            
            ## Delete my own entry from the map.
            ## Keep any other shared vertex I have in the map.
            #for i in range(0, len(self.incomplete_evp_elem)):
            #    elem_vertex_pair = (self.incomplete_evp_elem[i], self.incomplete_evp_vertex[i])
            #    self.local_shared_evp_to_elem_map[elem_vertex_pair] = self.local_shared_evp_to_elem_map[elem_vertex_pair][np.where(self.local_shared_evp_to_elem_map[elem_vertex_pair] != elem_vertex_pair[0])]
            #    self.local_shared_evp_to_vertex_map[elem_vertex_pair] = self.local_shared_evp_to_vertex_map[elem_vertex_pair][np.where(self.local_shared_evp_to_elem_map[elem_vertex_pair] != elem_vertex_pair[0])]
 
        if msh.gdim >= 1:
            
            self.log.write("info", "Computing local connectivity: Using edge centers")
            if msh.gdim == 2:
                num_edges = 4
                min_edges = 2
            else:
                num_edges = 12
                min_edges = 4

            # Allocate local dictionaries
            self.local_shared_eep_to_elem_map = {}
            self.local_shared_eep_to_edge_map = {}

            # For all elements, check all the edges
            for e in range(0, msh.nelv):
                    
                # For each edge, find any other edges that match the coordinates
                for edge in range(0, msh.edge_centers.shape[1]):
                    same_x =  np.isclose(msh.edge_centers[e, edge, 0],msh.edge_centers[:, :, 0], rtol=self.rtol)
                    same_y =  np.isclose(msh.edge_centers[e, edge, 1],msh.edge_centers[:, :, 1], rtol=self.rtol)
                    same_z =  np.isclose(msh.edge_centers[e, edge, 2],msh.edge_centers[:, :, 2], rtol=self.rtol)
                    same_edge = np.where(same_x & same_y & same_z)

                    matching_elem = same_edge[0]
                    matching_edge = same_edge[1]

                    self.local_shared_eep_to_elem_map[(e, edge)] = matching_elem
                    self.local_shared_eep_to_edge_map[(e, edge)] = matching_edge
            
            # For all edges where no match was found, indicate that they are "incomplete"
            # This means they can be boundary or shared with other ranks
            self.incomplete_eep_elem = []
            self.incomplete_eep_edge = []
            for elem_edge_pair in self.local_shared_eep_to_elem_map.keys():
                if len(self.local_shared_eep_to_elem_map[elem_edge_pair]) < min_edges: 
                    self.incomplete_eep_elem.append(elem_edge_pair[0])
                    self.incomplete_eep_edge.append(elem_edge_pair[1])

            ## Delete my own entry from the map.
            ## Keep any other shared edge I have in the map.
            #if msh.gdim == 2:
            #    for i in range(0, len(self.incomplete_eep_elem)):
            #        elem_edge_pair = (self.incomplete_eep_elem[i], self.incomplete_eep_edge[i])
            #        self.local_shared_eep_to_elem_map.pop(elem_edge_pair)
            #        self.local_shared_eep_to_edge_map.pop(elem_edge_pair) 
            #else:
            #    for i in range(0, len(self.incomplete_eep_elem)):
            #        elem_edge_pair = (self.incomplete_eep_elem[i], self.incomplete_eep_edge[i])
            #        self.local_shared_eep_to_elem_map[elem_edge_pair] = self.local_shared_eep_to_elem_map[elem_edge_pair][np.where(self.local_shared_eep_to_elem_map[elem_edge_pair] != elem_edge_pair[0])]
            #        self.local_shared_eep_to_edge_map[elem_edge_pair] = self.local_shared_eep_to_edge_map[elem_edge_pair][np.where(self.local_shared_eep_to_elem_map[elem_edge_pair] != elem_edge_pair[0])]        
         
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
                    
            ## Delete the unique element facet pairs from the local maps
            ## Since I just need to sum the values of the matching facet
            #for i in range(0, len(self.unique_efp_elem)):
            #    elem_facet_pair = (self.unique_efp_elem[i], self.unique_efp_facet[i])
            #    self.local_shared_efp_to_elem_map.pop(elem_facet_pair)
            #    self.local_shared_efp_to_facet_map.pop(elem_facet_pair)

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

        if msh.gdim >= 1:

            self.log.write("info", "Computing global connectivity: Using vertices")

            # Send incomplete vertices and their element and vertex id to all other ranks.
            destinations = [rank for rank in range(0, self.rt.comm.Get_size()) if rank != self.rt.comm.Get_rank()]

            local_incomplete_el_id = np.array(self.incomplete_evp_elem)
            local_incomplete_vertex_id = np.array(self.incomplete_evp_vertex)
            local_incomplete_vertex_coords = msh.vertices[self.incomplete_evp_elem, self.incomplete_evp_vertex]

            sources, source_incomplete_el_id = self.rt.all_to_all(destination=destinations, data=local_incomplete_el_id, dtype=local_incomplete_el_id.dtype)
            _ , source_incomplete_vertex_id = self.rt.all_to_all(destination=destinations, data=local_incomplete_vertex_id, dtype=local_incomplete_vertex_id.dtype)
            _, source_incomplete_vertex_coords = self.rt.all_to_all(destination=destinations, data=local_incomplete_vertex_coords, dtype=local_incomplete_vertex_coords.dtype)

            for i in range(0, len(source_incomplete_vertex_coords)):
                source_incomplete_vertex_coords[i] = source_incomplete_vertex_coords[i].reshape(-1, 3)

            # Create global dictionaries
            self.global_shared_evp_to_rank_map = {}
            self.global_shared_evp_to_elem_map = {}
            self.global_shared_evp_to_vertex_map = {}

            # Go through the data in each other rank.
            for source_idx, source_vc in enumerate(source_incomplete_vertex_coords):

                remove_pair_idx = []

                # Loop through all my own incomplete element vertex pairs
                for e_v_pair in range(0, len(self.incomplete_evp_elem)):

                    # Check where my incomplete vertex pair coordinates match with the incomplete ...
                    # ... vertex pair coordinates of the other rank
                    e = self.incomplete_evp_elem[e_v_pair]
                    vertex = self.incomplete_evp_vertex[e_v_pair]
                    same_x =  np.isclose(msh.vertices[e, vertex, 0],source_vc[:, 0], rtol=self.rtol)
                    same_y =  np.isclose(msh.vertices[e, vertex, 1],source_vc[:, 1], rtol=self.rtol)
                    same_z =  np.isclose(msh.vertices[e, vertex, 2],source_vc[:, 2], rtol=self.rtol)
                    same_vertex = np.where(same_x & same_y & same_z)
                        
                    # If we find a match assign it in the global dictionaries
                    if len(same_vertex[0]) > 0:
                        matching_id = same_vertex[0]
                        self.global_shared_evp_to_rank_map[(e, vertex)] = [sources[source_idx]]
                        self.global_shared_evp_to_elem_map[(e, vertex)] = source_incomplete_el_id[source_idx][matching_id]
                        self.global_shared_evp_to_vertex_map[(e, vertex)] = source_incomplete_vertex_id[source_idx][matching_id]
                        remove_pair_idx.append(e_v_pair)

                # If a match is found for an element vertex pair, remove it from the list of incomplete pairs
                remove_pair_idx = sorted(remove_pair_idx, reverse=True)
                for idx in remove_pair_idx:
                    self.incomplete_evp_elem.pop(idx)
                    self.incomplete_evp_vertex.pop(idx)

        if msh.gdim >= 1:

            self.log.write("info", "Computing global connectivity: Using edge centers")

            # Send incomplete edges and their element and edge id to all other ranks.
            destinations = [rank for rank in range(0, self.rt.comm.Get_size()) if rank != self.rt.comm.Get_rank()]

            local_incomplete_el_id = np.array(self.incomplete_eep_elem)
            local_incomplete_edge_id = np.array(self.incomplete_eep_edge)
            local_incomplete_edge_centers = msh.edge_centers[self.incomplete_eep_elem, self.incomplete_eep_edge]

            sources, source_incomplete_el_id = self.rt.all_to_all(destination=destinations, data=local_incomplete_el_id, dtype=local_incomplete_el_id.dtype)
            _ , source_incomplete_edge_id = self.rt.all_to_all(destination=destinations, data=local_incomplete_edge_id, dtype=local_incomplete_edge_id.dtype)
            _, source_incomplete_edge_centers = self.rt.all_to_all(destination=destinations, data=local_incomplete_edge_centers, dtype=local_incomplete_edge_centers.dtype)

            for i in range(0, len(source_incomplete_edge_centers)):
                source_incomplete_edge_centers[i] = source_incomplete_edge_centers[i].reshape(-1, 3)

            # Create global dictionaries
            self.global_shared_eep_to_rank_map = {}
            self.global_shared_eep_to_elem_map = {}
            self.global_shared_eep_to_edge_map = {}

            # Go through the data in each other rank.
            for source_idx, source_ec in enumerate(source_incomplete_edge_centers):

                remove_pair_idx = []

                # Loop through all my own incomplete element edge pairs
                for e_e_pair in range(0, len(self.incomplete_eep_elem)):

                    #check where my incomplete edge pair coordinates match with the incomplete ...
                    # ... edge pair coordinates of the other rank
                    e = self.incomplete_eep_elem[e_e_pair]
                    edge = self.incomplete_eep_edge[e_e_pair]
                    same_x =  np.isclose(msh.edge_centers[e, edge, 0],source_ec[:, 0], rtol=self.rtol)
                    same_y =  np.isclose(msh.edge_centers[e, edge, 1],source_ec[:, 1], rtol=self.rtol)
                    same_z =  np.isclose(msh.edge_centers[e, edge, 2],source_ec[:, 2], rtol=self.rtol)
                    same_edge = np.where(same_x & same_y & same_z)

                    # If we find a match assign it in the global dictionaries
                    if len(same_edge[0]) > 0:
                        matching_id = same_edge[0]
                        self.global_shared_eep_to_rank_map[(e, edge)] = [sources[source_idx]]
                        self.global_shared_eep_to_elem_map[(e, edge)] = source_incomplete_el_id[source_idx][matching_id]
                        self.global_shared_eep_to_edge_map[(e, edge)] = source_incomplete_edge_id[source_idx][matching_id]
                        remove_pair_idx.append(e_e_pair)
                
                # If a match is found for an element edge pair, remove it from the list of incomplete pairs
                remove_pair_idx = sorted(remove_pair_idx, reverse=True)
                for idx in remove_pair_idx:
                    self.incomplete_eep_elem.pop(idx)
                    self.incomplete_eep_edge.pop(idx)

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
                        self.global_shared_efp_to_rank_map[(e, facet)] = [sources[source_idx]]
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


    def get_multiplicity(self, msh: Mesh):
        """
        Computes the multiplicity of the elements in the mesh

        Parameters
        ----------
        msh : Mesh
        """

        self.multiplicity= np.ones_like(msh.x)

        if msh.gdim == 2:
            vertex_to_slice_map = vertex_to_slice_map_2d
            edge_to_slice_map = edge_to_slice_map_2d
        elif msh.gdim == 3:
            vertex_to_slice_map = vertex_to_slice_map_3d
            edge_to_slice_map = edge_to_slice_map_3d

        for e in range(0, msh.nelv):

            # Add number of vertices
            for vertex in range(0, msh.vertices.shape[1]):

                local_appearances = len(self.local_shared_evp_to_elem_map.get((e, vertex), []))
                global_appearances = len(self.global_shared_evp_to_elem_map.get((e, vertex), []))

                lz_index = vertex_to_slice_map[vertex][0] 
                ly_index = vertex_to_slice_map[vertex][1]
                lx_index = vertex_to_slice_map[vertex][2]

                self.multiplicity[e, lz_index, ly_index, lx_index] = local_appearances + global_appearances

            
            # Add number of edges
            for edge in range(0, msh.edge_centers.shape[1]):

                local_appearances = len(self.local_shared_eep_to_elem_map.get((e, edge), []))
                global_appearances = len(self.global_shared_eep_to_elem_map.get((e, edge), []))

                lz_index = edge_to_slice_map[edge][0] 
                ly_index = edge_to_slice_map[edge][1]
                lx_index = edge_to_slice_map[edge][2]

                # Exclude vertices
                if lz_index == slice(None):
                    lz_index = slice(1, -1)
                if ly_index == slice(None):
                    ly_index = slice(1, -1)
                if lx_index == slice(None):
                    lx_index = slice(1, -1)

                self.multiplicity[e, lz_index, ly_index, lx_index] = local_appearances + global_appearances

            if msh.gdim == 3:

                # Add number of facets
                for facet in range(0, 6):

                    local_appearances = len(self.local_shared_efp_to_elem_map.get((e, facet), []))
                    global_appearances = len(self.global_shared_efp_to_elem_map.get((e, facet), []))

                    lz_index = facet_to_slice_map[facet][0] 
                    ly_index = facet_to_slice_map[facet][1]
                    lx_index = facet_to_slice_map[facet][2]
                
                    # Exclude edges
                    if lz_index == slice(None):
                        lz_index = slice(1, -1)
                    if ly_index == slice(None):
                        ly_index = slice(1, -1)
                    if lx_index == slice(None):
                        lx_index = slice(1, -1)

                    self.multiplicity[e, lz_index, ly_index, lx_index] = local_appearances + global_appearances


    def dssum_local(self, field: np.ndarray = None, msh:  Mesh = None, coef: Coef = None):

        self.log.write("info", "Computing local dssum")
        self.log.tic()

        avrg_field = np.copy(field)

        if msh.gdim == 2:
            vertex_to_slice_map = vertex_to_slice_map_2d
            edge_to_slice_map = edge_to_slice_map_2d
        elif msh.gdim == 3:
            vertex_to_slice_map = vertex_to_slice_map_3d
            edge_to_slice_map = edge_to_slice_map_3d


        self.log.write("info", "Adding vertices")
        for e in range(0, msh.nelv):

            # Vertex data is pointwise and can be summed directly 
            for vertex in range(0, msh.vertices.shape[1]):

                if (e, vertex) in self.local_shared_evp_to_elem_map.keys():

                    # Get the data from other elements
                    shared_elements_ = list(self.local_shared_evp_to_elem_map[(e, vertex)])
                    shared_vertices_ = list(self.local_shared_evp_to_vertex_map[(e, vertex)])

                    # Filter out my own element from the list
                    shared_elements = [shared_elements_[ii] for ii in range(0, len(shared_elements_)) if shared_elements_[ii] != e]                    
                    shared_vertices = [shared_vertices_[ii] for ii in range(0, len(shared_elements_)) if shared_elements_[ii] != e]

                    if shared_vertices == []:
                        continue

                    # Get the vertex data from the other elements of the field.
                    shared_vertex_data = vd(field=field, elem=shared_elements, vertex=shared_vertices)

                    # Get the vertex location on my own elemenet
                    lz_index = vertex_to_slice_map[vertex][0] 
                    ly_index = vertex_to_slice_map[vertex][1]
                    lx_index = vertex_to_slice_map[vertex][2]

                    avrg_field[e, lz_index, ly_index, lx_index] += np.sum(shared_vertex_data)

        self.log.write("info", "Adding edges")
        for e in range(0, msh.nelv):
            # Edge data is provided as a line that might be flipped, we must compare values of the mesh
            for edge in range(0, msh.edge_centers.shape[1]):
                
                if (e, edge) in self.local_shared_eep_to_elem_map.keys():
                    
                    # Get the data from other elements
                    shared_elements_ = list(self.local_shared_eep_to_elem_map[(e, edge)])
                    shared_edges_ = list(self.local_shared_eep_to_edge_map[(e, edge)])
                    
                    # Filter out my own element from the list
                    shared_elements = [shared_elements_[ii] for ii in range(0, len(shared_elements_)) if shared_elements_[ii] != e]                    
                    shared_edges = [shared_edges_[ii] for ii in range(0, len(shared_elements_)) if shared_elements_[ii] != e]                    

                    if shared_edges == []:
                        continue

                    # Get the shared edge coordinates from the other elements
                    shared_edge_coord_x = ed(field=msh.x, elem=shared_elements, edge=shared_edges)
                    shared_edge_coord_y = ed(field=msh.y, elem=shared_elements, edge=shared_edges)
                    shared_edge_coord_z = ed(field=msh.z, elem=shared_elements, edge=shared_edges)

                    # Get the shared edge data from the other elements of the field.
                    shared_edge_data = ed(field=field, elem=shared_elements, edge=shared_edges)

                    # Get the edge location on my own elemenet
                    lz_index = edge_to_slice_map[edge][0]
                    ly_index = edge_to_slice_map[edge][1]
                    lx_index = edge_to_slice_map[edge][2]
 
                    # Get my own edge data and coordinates
                    my_edge_coord_x = msh.x[e, lz_index, ly_index, lx_index]
                    my_edge_coord_y = msh.y[e, lz_index, ly_index, lx_index]
                    my_edge_coord_z = msh.z[e, lz_index, ly_index, lx_index]
                    my_edge_data = np.copy(field[e, lz_index, ly_index, lx_index])

                    # Compare coordinates excluding the vertices
                    # For each of my data points
                    for edge_point in range(1, my_edge_coord_x.shape[0]-1):
                        edge_point_x = my_edge_coord_x[edge_point]
                        edge_point_y = my_edge_coord_y[edge_point]
                        edge_point_z = my_edge_coord_z[edge_point]

                        # For each element that shares edge data
                        for sharing_elem in range(0, len(shared_elements)):
                            sharing_elem_edge_coord_x = shared_edge_coord_x[sharing_elem]
                            sharing_elem_edge_coord_y = shared_edge_coord_y[sharing_elem]
                            sharing_elem_edge_coord_z = shared_edge_coord_z[sharing_elem]

                            # Compare
                            same_x = np.isclose(edge_point_x, sharing_elem_edge_coord_x, rtol=self.rtol)
                            same_y = np.isclose(edge_point_y, sharing_elem_edge_coord_y, rtol=self.rtol)
                            same_z = np.isclose(edge_point_z, sharing_elem_edge_coord_z, rtol=self.rtol)
                            same_edge_point = np.where(same_x & same_y & same_z)

                            # Sum where a match is found
                            if len(same_edge_point[0]) > 0:
                                my_edge_data[edge_point] += shared_edge_data[sharing_elem][same_edge_point]

                    # Do not assing at the vertices
                    if lz_index == slice(None):
                        lz_index = slice(1, -1)
                    if ly_index == slice(None):
                        ly_index = slice(1, -1)
                    if lx_index == slice(None):
                        lx_index = slice(1, -1)
                    slice_copy = slice(1, -1)
                    avrg_field[e, lz_index, ly_index, lx_index] = np.copy(my_edge_data[slice_copy])

        if msh.gdim == 3:
            self.log.write("info", "Adding faces")
            for e in range(0, msh.nelv):

                # Facet data might be flipper or rotated so better check coordinates
                for facet in range(0, 6):

                    if (e, facet) in self.local_shared_efp_to_elem_map.keys():

                        # Get the data from other elements
                        shared_elements_ = list(self.local_shared_efp_to_elem_map[(e, facet)])
                        shared_facets_ = list(self.local_shared_efp_to_facet_map[(e, facet)])

                        # Filter out my own element from the list
                        shared_elements = [shared_elements_[ii] for ii in range(0, len(shared_elements_)) if shared_elements_[ii] != e]                    
                        shared_facets = [shared_facets_[ii] for ii in range(0, len(shared_elements_)) if shared_elements_[ii] != e]                    

                        if shared_facets == []:
                            continue

                        # Get the shared facet coordinates from the other elements
                        shared_facet_coord_x = fd(field=msh.x, elem=shared_elements, facet=shared_facets)
                        shared_facet_coord_y = fd(field=msh.y, elem=shared_elements, facet=shared_facets)
                        shared_facet_coord_z = fd(field=msh.z, elem=shared_elements, facet=shared_facets)

                        # Get the shared facet data from the other elements of the field.
                        shared_facet_data = fd(field=field, elem=shared_elements, facet=shared_facets)

                        # Get the facet location on my own elemenet
                        lz_index = facet_to_slice_map[facet][0]
                        ly_index = facet_to_slice_map[facet][1]
                        lx_index = facet_to_slice_map[facet][2]

                        # Get my own facet data and coordinates
                        my_facet_coord_x = msh.x[e, lz_index, ly_index, lx_index]
                        my_facet_coord_y = msh.y[e, lz_index, ly_index, lx_index]
                        my_facet_coord_z = msh.z[e, lz_index, ly_index, lx_index]
                        my_facet_data = np.copy(field[e, lz_index, ly_index, lx_index])

                        # Compare coordinates excluding the edges
                        # For each of my data points
                        for facet_point_j in range(1, my_facet_coord_x.shape[0]-1):
                            for facet_point_i in range(1, my_facet_coord_x.shape[1]-1):
                                facet_point_x = my_facet_coord_x[facet_point_j, facet_point_i]
                                facet_point_y = my_facet_coord_y[facet_point_j, facet_point_i]
                                facet_point_z = my_facet_coord_z[facet_point_j, facet_point_i]

                                # For each element that shares facet data
                                for sharing_elem in range(0, len(shared_elements)):
                                    sharing_elem_facet_coord_x = shared_facet_coord_x[sharing_elem]
                                    sharing_elem_facet_coord_y = shared_facet_coord_y[sharing_elem]
                                    sharing_elem_facet_coord_z = shared_facet_coord_z[sharing_elem]

                                    # Compare
                                    same_x = np.isclose(facet_point_x, sharing_elem_facet_coord_x, rtol=self.rtol)
                                    same_y = np.isclose(facet_point_y, sharing_elem_facet_coord_y, rtol=self.rtol)
                                    same_z = np.isclose(facet_point_z, sharing_elem_facet_coord_z, rtol=self.rtol)
                                    same_facet_point = np.where(same_x & same_y & same_z)

                                    # Sum where a match is found
                                    if len(same_facet_point[0]) > 0:
                                        my_facet_data[facet_point_j, facet_point_i] += shared_facet_data[sharing_elem][same_facet_point]
                        
                        # Do not assing at the edges
                        if lz_index == slice(None):
                            lz_index = slice(1, -1)
                        if ly_index == slice(None):
                            ly_index = slice(1, -1)
                        if lx_index == slice(None):
                            lx_index = slice(1, -1)
                        slice_copy = slice(1, -1)
                        avrg_field[e, lz_index, ly_index, lx_index] = np.copy(my_facet_data[slice_copy, slice_copy])

        # Divide by multiplicity    
        #avrg_field = avrg_field / self.multiplicity

        self.log.write("info", "Local dssum computed")
        self.log.toc()

        return avrg_field

    def dssum_global(self, comm, summed_field: np.ndarray = None, field: np.ndarray = None, msh:  Mesh = None, coef: Coef = None):

        self.log.write("info", "Computing global dssum")
        self.log.tic()

        avrg_field = np.copy(summed_field)

        if msh.gdim == 2:
            vertex_to_slice_map = vertex_to_slice_map_2d
            edge_to_slice_map = edge_to_slice_map_2d
        elif msh.gdim == 3:
            vertex_to_slice_map = vertex_to_slice_map_3d
            edge_to_slice_map = edge_to_slice_map_3d

        # Prepare vertices to send to other ranks:
        vertex_send_buff = {}
        for e in range(0, msh.nelv):
            
            # Vertex data is pointwise and can be summed directly 
            for vertex in range(0, msh.vertices.shape[1]):


                if (e, vertex) in self.global_shared_evp_to_elem_map.keys():

                    # Check which other rank has this vertex
                    shared_ranks = list(self.global_shared_evp_to_rank_map[(e, vertex)])

                    # Get the vertex data from the other elements of the field.
                    my_vertex_coord_x = vd(field=msh.x, elem=e, vertex=vertex)
                    my_vertex_coord_y = vd(field=msh.y, elem=e, vertex=vertex)
                    my_vertex_coord_z = vd(field=msh.z, elem=e, vertex=vertex)
                    my_vertex_data = vd(field=field, elem=e, vertex=vertex)

                    for rank in shared_ranks:
                            
                        # Send the vertex data to the other rank
                        if rank not in vertex_send_buff.keys():
                            vertex_send_buff[rank] = {}
                            vertex_send_buff[rank]['e'] = []
                            vertex_send_buff[rank]['vertex'] = []
                            vertex_send_buff[rank]['x_coords'] = []
                            vertex_send_buff[rank]['y_coords'] = []
                            vertex_send_buff[rank]['z_coords'] = []
                            vertex_send_buff[rank]['data'] = []
    
                        vertex_send_buff[rank]['e'].append(e)
                        vertex_send_buff[rank]['vertex'].append(vertex)
                        vertex_send_buff[rank]['x_coords'].append(my_vertex_coord_x)
                        vertex_send_buff[rank]['y_coords'].append(my_vertex_coord_y)
                        vertex_send_buff[rank]['z_coords'].append(my_vertex_coord_z)
                        vertex_send_buff[rank]['data'].append(my_vertex_data)


        # Prepare edges to send to other ranks:
        edge_send_buff = {}
        for e in range(0, msh.nelv):

            # Edge data is provided as a line that might be flipped, we must compare values of the mesh
            for edge in range(0, msh.edge_centers.shape[1]):

                if (e, edge) in self.global_shared_eep_to_elem_map.keys():

                    # Check which other rank has this edge
                    shared_ranks = list(self.global_shared_eep_to_rank_map[(e, edge)])

                    # Get the edge data from the other elements of the field.
                    my_edge_coord_x = ed(field=msh.x, elem=e, edge=edge)
                    my_edge_coord_y = ed(field=msh.y, elem=e, edge=edge)
                    my_edge_coord_z = ed(field=msh.z, elem=e, edge=edge)
                    my_edge_data = ed(field=field, elem=e, edge=edge)

                    for rank in shared_ranks:
                            
                        # Send the edge data to the other rank
                        if rank not in edge_send_buff.keys():
                            edge_send_buff[rank] = {}
                            edge_send_buff[rank]['e'] = []
                            edge_send_buff[rank]['edge'] = []
                            edge_send_buff[rank]['x_coords'] = []
                            edge_send_buff[rank]['y_coords'] = []
                            edge_send_buff[rank]['z_coords'] = []
                            edge_send_buff[rank]['data'] = []
    
                        edge_send_buff[rank]['e'].append(e)
                        edge_send_buff[rank]['edge'].append(edge)
                        edge_send_buff[rank]['x_coords'].append(my_edge_coord_x)
                        edge_send_buff[rank]['y_coords'].append(my_edge_coord_y)
                        edge_send_buff[rank]['z_coords'].append(my_edge_coord_z)
                        edge_send_buff[rank]['data'].append(my_edge_data)


        # Prepare the facets to send to other ranks:
        if msh.gdim == 3:
            facet_send_buff = {}
            for e in range(0, msh.nelv):

                # Facet data might be flipper or rotated so better check coordinates
                for facet in range(0, 6):

                    if (e, facet) in self.global_shared_efp_to_elem_map.keys():

                        # Check which other rank has this facet
                        shared_ranks = list(self.global_shared_efp_to_rank_map[(e, facet)])

                        # Get the facet data from the other elements of the field.
                        my_facet_coord_x = fd(field=msh.x, elem=e, facet=facet)
                        my_facet_coord_y = fd(field=msh.y, elem=e, facet=facet)
                        my_facet_coord_z = fd(field=msh.z, elem=e, facet=facet)
                        my_facet_data = fd(field=field, elem=e, facet=facet)

                        for rank in shared_ranks:
                                
                            # Send the facet data to the other rank
                            if rank not in facet_send_buff.keys():
                                facet_send_buff[rank] = {}
                                facet_send_buff[rank]['e'] = []
                                facet_send_buff[rank]['facet'] = []
                                facet_send_buff[rank]['x_coords'] = []
                                facet_send_buff[rank]['y_coords'] = []
                                facet_send_buff[rank]['z_coords'] = []
                                facet_send_buff[rank]['data'] = []
        
                            facet_send_buff[rank]['e'].append(e)
                            facet_send_buff[rank]['facet'].append(facet)
                            facet_send_buff[rank]['x_coords'].append(my_facet_coord_x)
                            facet_send_buff[rank]['y_coords'].append(my_facet_coord_y)
                            facet_send_buff[rank]['z_coords'].append(my_facet_coord_z)
                            facet_send_buff[rank]['data'].append(my_facet_data)

        # Now send the vertices
        destinations = [rank for rank in vertex_send_buff.keys()]
        local_vertex_el_id = [np.array(vertex_send_buff[rank]['e']) for rank in destinations]
        local_vertex_id = [np.array(vertex_send_buff[rank]['vertex']) for rank in destinations]
        local_vertex_x_coords = [np.array(vertex_send_buff[rank]['x_coords']) for rank in destinations]
        local_vertex_y_coords = [np.array(vertex_send_buff[rank]['y_coords']) for rank in destinations]
        local_vertex_z_coords = [np.array(vertex_send_buff[rank]['z_coords']) for rank in destinations]
        local_vertex_data = [np.array(vertex_send_buff[rank]['data']) for rank in destinations]
        vertex_sources, source_vertex_el_id = self.rt.all_to_all(destination=destinations, data=local_vertex_el_id, dtype=local_vertex_el_id[0].dtype)
        _ , source_vertex_id = self.rt.all_to_all(destination=destinations, data=local_vertex_id, dtype=local_vertex_id[0].dtype)
        _, source_vertex_x_coords = self.rt.all_to_all(destination=destinations, data=local_vertex_x_coords, dtype=local_vertex_x_coords[0].dtype)
        _, source_vertex_y_coords = self.rt.all_to_all(destination=destinations, data=local_vertex_y_coords, dtype=local_vertex_y_coords[0].dtype)
        _, source_vertex_z_coords = self.rt.all_to_all(destination=destinations, data=local_vertex_z_coords, dtype=local_vertex_z_coords[0].dtype)
        _, source_vertex_data = self.rt.all_to_all(destination=destinations, data=local_vertex_data, dtype=local_vertex_data[0].dtype)

        for i in range(0, len(source_vertex_x_coords)):
            # No need to reshape the vertex data
            pass 

        # Now send the edges
        destinations = [rank for rank in edge_send_buff.keys()]
        local_edge_el_id = [np.array(edge_send_buff[rank]['e']) for rank in destinations]
        local_edge_id = [np.array(edge_send_buff[rank]['edge']) for rank in destinations]
        local_edge_x_coords = [np.array(edge_send_buff[rank]['x_coords']) for rank in destinations]
        local_edge_y_coords = [np.array(edge_send_buff[rank]['y_coords']) for rank in destinations]
        local_edge_z_coords = [np.array(edge_send_buff[rank]['z_coords']) for rank in destinations]
        local_edge_data = [np.array(edge_send_buff[rank]['data']) for rank in destinations]
        edges_sources, source_edge_el_id = self.rt.all_to_all(destination=destinations, data=local_edge_el_id, dtype=local_edge_el_id[0].dtype)
        _ , source_edge_id = self.rt.all_to_all(destination=destinations, data=local_edge_id, dtype=local_edge_id[0].dtype)
        _, source_edge_x_coords = self.rt.all_to_all(destination=destinations, data=local_edge_x_coords, dtype=local_edge_x_coords[0].dtype)
        _, source_edge_y_coords = self.rt.all_to_all(destination=destinations, data=local_edge_y_coords, dtype=local_edge_y_coords[0].dtype)
        _, source_edge_z_coords = self.rt.all_to_all(destination=destinations, data=local_edge_z_coords, dtype=local_edge_z_coords[0].dtype)
        _, source_edge_data = self.rt.all_to_all(destination=destinations, data=local_edge_data, dtype=local_edge_data[0].dtype)

        for i in range(0, len(source_edge_x_coords)):
            source_edge_x_coords[i] = source_edge_x_coords[i].reshape(-1, msh.lx)
            source_edge_y_coords[i] = source_edge_y_coords[i].reshape(-1, msh.lx)
            source_edge_z_coords[i] = source_edge_z_coords[i].reshape(-1, msh.lx)
            source_edge_data[i] = source_edge_data[i].reshape(-1, msh.lx)

        # Now send the facets
        if msh.gdim == 3:
            destinations = [rank for rank in facet_send_buff.keys()]
            local_facet_el_id = [np.array(facet_send_buff[rank]['e']) for rank in destinations]
            local_facet_id = [np.array(facet_send_buff[rank]['facet']) for rank in destinations]
            local_facet_x_coords = [np.array(facet_send_buff[rank]['x_coords']) for rank in destinations]
            local_facet_y_coords = [np.array(facet_send_buff[rank]['y_coords']) for rank in destinations]
            local_facet_z_coords = [np.array(facet_send_buff[rank]['z_coords']) for rank in destinations]
            local_facet_data = [np.array(facet_send_buff[rank]['data']) for rank in destinations]
            facet_sources, source_facet_el_id = self.rt.all_to_all(destination=destinations, data=local_facet_el_id, dtype=local_facet_el_id[0].dtype)
            _ , source_facet_id = self.rt.all_to_all(destination=destinations, data=local_facet_id, dtype=local_facet_id[0].dtype)
            _, source_facet_x_coords = self.rt.all_to_all(destination=destinations, data=local_facet_x_coords, dtype=local_facet_x_coords[0].dtype)
            _, source_facet_y_coords = self.rt.all_to_all(destination=destinations, data=local_facet_y_coords, dtype=local_facet_y_coords[0].dtype)
            _, source_facet_z_coords = self.rt.all_to_all(destination=destinations, data=local_facet_z_coords, dtype=local_facet_z_coords[0].dtype)
            _, source_facet_data = self.rt.all_to_all(destination=destinations, data=local_facet_data, dtype=local_facet_data[0].dtype)

            for i in range(0, len(source_facet_x_coords)):
                source_facet_x_coords[i] = source_facet_x_coords[i].reshape(-1, msh.ly, msh.lx)
                source_facet_y_coords[i] = source_facet_y_coords[i].reshape(-1, msh.ly, msh.lx)
                source_facet_z_coords[i] = source_facet_z_coords[i].reshape(-1, msh.ly, msh.lx)
                source_facet_data

        # Summ vertices:
        for e in range(0, msh.nelv):
            
            # Vertex data is pointwise and can be summed directly 
            for vertex in range(0, msh.vertices.shape[1]):

                if (e, vertex) in self.global_shared_evp_to_elem_map.keys():

                    # Check which other rank has this vertex
                    shared_ranks = list(self.global_shared_evp_to_rank_map[(e, vertex)])
                    shared_elements = list(self.global_shared_evp_to_elem_map[(e, vertex)])
                    shared_vertices = list(self.global_shared_evp_to_vertex_map[(e, vertex)])

                    # Get the vertex data from the different ranks
                    for sv in range(0, len(shared_vertices)):

                        source_index = shared_ranks.index(shared_ranks[0]) # This one should be come a list of the same size

                        # Get the data from this source
                        shared_vertex_el_id = source_vertex_el_id[source_index]
                        shared_vertex_id = source_vertex_id[source_index]
                        shared_vertex_coord_x = source_vertex_x_coords[source_index]
                        shared_vertex_coord_y = source_vertex_y_coords[source_index]
                        shared_vertex_coord_z = source_vertex_z_coords[source_index]
                        shared_vertex_data = source_vertex_data[source_index]
                        
                        # find the data that matches the element and vertex id dictionary
                        el = shared_elements[sv]
                        vertex_id = shared_vertices[sv]
                        same_el = shared_vertex_el_id == el
                        same_vertex = shared_vertex_id == vertex_id
                        matching_index = np.where(same_el & same_vertex)

                        matching_vertex_coord_x = shared_vertex_coord_x[matching_index]
                        matching_vertex_coord_y = shared_vertex_coord_y[matching_index]
                        matching_vertex_coord_z = shared_vertex_coord_z[matching_index]
                        matching_vertex_data    = shared_vertex_data[matching_index]


                        # Get my own values (It is not really needed for the vertices)
                        my_vertex_coord_x = vd(field=msh.x, elem=e, vertex=vertex)
                        my_vertex_coord_y = vd(field=msh.y, elem=e, vertex=vertex)
                        my_vertex_coord_z = vd(field=msh.z, elem=e, vertex=vertex)
                        my_vertex_data = np.copy(vd(field=field, elem=e, vertex=vertex))

                        # Get the vertex location on my own elemenet
                        lz_index = vertex_to_slice_map[vertex][0] 
                        ly_index = vertex_to_slice_map[vertex][1]
                        lx_index = vertex_to_slice_map[vertex][2]

                        # Add the data from this rank, element, vertex triad.
                        avrg_field[e, lz_index, ly_index, lx_index] += matching_vertex_data
                        

        # Summ edges:
        for e in range(0, msh.nelv):

            # Edge data is provided as a line that might be flipped, we must compare values of the mesh
            for edge in range(0, msh.edge_centers.shape[1]):

                if (e, edge) in self.global_shared_eep_to_elem_map.keys():

                    # Check which other rank has this edge
                    shared_ranks = list(self.global_shared_eep_to_rank_map[(e, edge)])
                    shared_elements = list(self.global_shared_eep_to_elem_map[(e, edge)])
                    shared_edges = list(self.global_shared_eep_to_edge_map[(e, edge)])

                    # Get the edge data from the different ranks
                    for se in range(0, len(shared_edges)):

                        source_index = shared_ranks.index(shared_ranks[0])

                        # Get the data from this source
                        shared_edge_el_id = source_edge_el_id[source_index]
                        shared_edge_id = source_edge_id[source_index]
                        shared_edge_coord_x = source_edge_x_coords[source_index]
                        shared_edge_coord_y = source_edge_y_coords[source_index]
                        shared_edge_coord_z = source_edge_z_coords[source_index]
                        shared_edge_data = source_edge_data[source_index]

                        # find the data that matches the element and edge id dictionary
                        el = shared_elements[se]
                        edge_id = shared_edges[se]
                        same_el = shared_edge_el_id == el
                        same_edge = shared_edge_id == edge_id
                        matching_index = np.where(same_el & same_edge)

                        matching_edge_coord_x = shared_edge_coord_x[matching_index]
                        matching_edge_coord_y = shared_edge_coord_y[matching_index]
                        matching_edge_coord_z = shared_edge_coord_z[matching_index]
                        matching_edge_data    = shared_edge_data[matching_index]

                        # Get the edge location on my own elemenet
                        lz_index = edge_to_slice_map[edge][0]
                        ly_index = edge_to_slice_map[edge][1]
                        lx_index = edge_to_slice_map[edge][2]
    
                        # Get my own edge data and coordinates
                        my_edge_coord_x = msh.x[e, lz_index, ly_index, lx_index]
                        my_edge_coord_y = msh.y[e, lz_index, ly_index, lx_index]
                        my_edge_coord_z = msh.z[e, lz_index, ly_index, lx_index]
                        my_edge_data = np.copy(summed_field[e, lz_index, ly_index, lx_index])

                        # Compare coordinates excluding the vertices
                        # For each of my data points
                        for edge_point in range(1, my_edge_coord_x.shape[0]-1):
                            edge_point_x = my_edge_coord_x[edge_point]
                            edge_point_y = my_edge_coord_y[edge_point]
                            edge_point_z = my_edge_coord_z[edge_point]
                            
                            # Compare
                            same_x = np.isclose(edge_point_x, matching_edge_coord_x, rtol=self.rtol)
                            same_y = np.isclose(edge_point_y, matching_edge_coord_y, rtol=self.rtol)
                            same_z = np.isclose(edge_point_z, matching_edge_coord_z, rtol=self.rtol)
                            same_edge_point = np.where(same_x & same_y & same_z)

                            # Sum where a match is found
                            if len(same_edge_point[0]) > 0:
                                my_edge_data[edge_point] += matching_edge_data[same_edge_point]

                        # Do not assing at the vertices
                        if lz_index == slice(None):
                            lz_index = slice(1, -1)
                        if ly_index == slice(None):
                            ly_index = slice(1, -1)
                        if lx_index == slice(None):
                            lx_index = slice(1, -1)
                        slice_copy = slice(1, -1)
                        avrg_field[e, lz_index, ly_index, lx_index] = np.copy(my_edge_data[slice_copy])

        # Divide by multiplicity    
        avrg_field = avrg_field / self.multiplicity

        self.log.write("info", "Local dssum computed")
        self.log.toc()

        return avrg_field
                     


