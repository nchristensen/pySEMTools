'''Contains method for slicing elements'''

from typing import Union
import numpy as np


facet_to_slice_map = { "facet" : ('lz', 'ly', 'lx') }
facet_to_slice_map[0] = (slice(None), slice(None), 0) # Equivalent to [:, :, 0]
facet_to_slice_map[1] = (slice(None), slice(None), -1) # Equivalent to [:, :, -1]
facet_to_slice_map[2] = (slice(None), 0, slice(None)) # Equivalent to [:, 0, :]
facet_to_slice_map[3] = (slice(None), -1, slice(None)) # Equivalent to [:, -1, :]
facet_to_slice_map[4] = (0, slice(None), slice(None)) # Equivalent to [0, :, :]
facet_to_slice_map[5] = (-1, slice(None), slice(None)) # Equivalent to [-1, :, :]

vertex_to_slice_map_3d = { "vertex" : ('lz', 'ly', 'lx') }
vertex_to_slice_map_3d[0] = (0, 0, 0) # Equivalent to [0, 0, 0]
vertex_to_slice_map_3d[1] = (0, 0, -1) # Equivalent to [0, 0, -1]
vertex_to_slice_map_3d[2] = (0, -1, 0) # Equivalent to [0, -1, 0]
vertex_to_slice_map_3d[3] = (0, -1, -1) # Equivalent to [0, -1, -1]
vertex_to_slice_map_3d[4] = (-1, 0, 0) # Equivalent to [-1, 0, 0]
vertex_to_slice_map_3d[5] = (-1, 0, -1) # Equivalent to [-1, 0, -1]
vertex_to_slice_map_3d[6] = (-1, -1, 0) # Equivalent to [-1, -1, 0]
vertex_to_slice_map_3d[7] = (-1, -1, -1) # Equivalent to [-1, -1, -1]

vertex_to_slice_map_2d = { "vertex" : ('lz', 'ly', 'lx') }
vertex_to_slice_map_2d[0] = (0, 0, 0) # Equivalent to [0, 0, 0]
vertex_to_slice_map_2d[1] = (0, 0, -1) # Equivalent to [0, 0, -1]
vertex_to_slice_map_2d[2] = (0, -1, 0) # Equivalent to [0, -1, 0]
vertex_to_slice_map_2d[3] = (0, -1, -1) # Equivalent to [0, -1, -1]

def fetch_elem_vertex_data(field: np.ndarray = None, elem: Union[int, list[int]] = None,vertex: Union[int, list[int]] = None):
    """

    Fetch the data from the field for the given element and vertex pair

    Parameters
    ----------
    field : np.ndarray
        The field data  to be sliced.
        The field shape should be (nelv, lz, ly, lx) for 3D and 2D. 2D has lz = 1.
    
    elem : Union[int, list[int]]
        The element index or list of element indices
    
    vertex : Union[int, list[int]]
        The vertex index or list of vertex indices
    
    Notes
    -----

    The restriction is that if vertex is a list, elem must be a list of the same length.
    This because then we retrieve the data for vertex[i] from elem[i].

    If vertex is an integer, then that means that we will retieve the same vertex from all.
    In this case:
    elem can be an interger, which returns the data for the elem and vertex pair.
    elem can be a list of integers, which returns the data for the list of elem and the same vertex.
    elem can be None, which returns the data for all elements and the same vertex.

    Returns
    -------
    np.ndarray
        The data for the given element and vertex pair
    """
    
    if field.shape[1] > 1:
        vertex_to_slice_map = vertex_to_slice_map_3d
    else:
        vertex_to_slice_map = vertex_to_slice_map_2d

    # Get the slice tuple for each element in the list
    if isinstance(elem, list):
        elem_index = [e for e in elem]
    elif isinstance(elem, int):
        elem_index =  elem
    else:
        elem_index = slice(None)

    if isinstance(vertex, list):
        lz_index = [vertex_to_slice_map[v][0] for v in vertex]
        ly_index = [vertex_to_slice_map[v][1] for v in vertex]
        lx_index = [vertex_to_slice_map[v][2] for v in vertex]
    elif isinstance(vertex, int):
        lz_index = vertex_to_slice_map[vertex][0]
        ly_index = vertex_to_slice_map[vertex][1]
        lx_index = vertex_to_slice_map[vertex][2]
    else:
        raise ValueError("Vertex must be an integer or a list of integers")

    # Retrieve the data if inputs are list of corresponding element and facet pairs
    if isinstance(elem, list) and isinstance(vertex, list):
        # Create slices
        slices = [(elem_index[i], lz_index[i], ly_index[i], lx_index[i]) for i in range(len(elem_index))]
        # Retrieve the data
        vertex_data = np.array([field[s] for s in slices])

    # Retrieve the same facet from one element, elements in a list, or all elements
    elif (isinstance(elem, list) or isinstance(elem, int) or isinstance(elem, type(None))) and isinstance(vertex, int): 
        slices = (elem_index, lz_index, ly_index, lx_index)
        vertex_data = field[slices]
    
    else:
        raise TypeError("Invalid input: If vertex is a list, elem must be a list of the same length")

    return vertex_data



def fetch_elem_facet_data(field: np.ndarray = None, elem: Union[int, list[int]] = None, facet: Union[int, list[int]] = None):
    """
    Fetch the data from the field for the given element and facet pair

    Parameters
    ----------
    field : np.ndarray
        The field data  to be sliced.
        The field shape should be (nelv, lz, ly, lx) for 3D and 2D.
    
    elem : Union[int, list[int]]
        The element index or list of element indices
    
    facet : Union[int, list[int]]
        The facet index or list of facet indices
    
    Notes
    -----
    The restriction is that if facet is a list, elem must be a list of the same length.
    This because then we retrieve the data for facet[i] from elem[i].

    If facet is an integer, then that means that we will retieve the same facet from all.
    In this case: 
    elem can be an interger, which returns the data for the elem and facet pair.
    elem can be a list of integers, which returns the data for the list of elem and the same facet.
    elem can be None, which returns the data for all elements and the same facet.
    
    Returns
    -------
    np.ndarray
        The data for the given element and facet pair
    """

    # Get the slice tuple for each element in the list
    if isinstance(elem, list):
        elem_index = [e for e in elem]
    elif isinstance(elem, int):
        elem_index =  elem
    else:
        elem_index = slice(None)

    if isinstance(facet, list):
        lz_index = [facet_to_slice_map[f][0] for f in facet]
        ly_index = [facet_to_slice_map[f][1] for f in facet]
        lx_index = [facet_to_slice_map[f][2] for f in facet]
    elif isinstance(facet, int):
        lz_index = facet_to_slice_map[facet][0]
        ly_index = facet_to_slice_map[facet][1]
        lx_index = facet_to_slice_map[facet][2]
    else:
        raise ValueError("Facet must be an integer or a list of integers")

    # Retrieve the data if inputs are list of corresponding element and facet pairs
    if isinstance(elem, list) and isinstance(facet, list):
        # Create slices
        slices = [(elem_index[i], lz_index[i], ly_index[i], lx_index[i]) for i in range(len(elem_index))]
        # Retrieve the data
        facet_data = np.array([field[s] for s in slices])

    # Retrieve the same facet from one element, elements in a list, or all elements
    elif (isinstance(elem, list) or isinstance(elem, int) or isinstance(elem, type(None))) and isinstance(facet, int): 
        slices = (elem_index, lz_index, ly_index, lx_index)
        facet_data = field[slices]
    
    else:
        raise TypeError("Invalid input: If facet is a list, elem must be a list of the same length")

    return facet_data