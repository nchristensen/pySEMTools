"""'Contains the probes class"""

import json
import csv
import numpy as np
import os
from ..io.ppymech.neksuite import preadnek, pynekread
from .interpolator import Interpolator
from ..monitoring.logger import Logger
from typing import Union
from ..datatypes.msh import Mesh

NoneType = type(None)


class Probes:
    """
    Class to interpolate fields at probes from a SEM mesh.

    Main interpolation class. This works in parallel, however, the probes are held at rank 0.
    therefore if initializing from file, the probes file should be read at rank 0 and then
    scattered to all ranks.

    If the probes are passed as an argument, they should be passed to all ranks, however only the
    data in rank 0 will be relevant (an scatter to all other ranks). See example below to observe
    how to avoid unnecessary replication of data in all ranks when passing probes as argument.

    Parameters
    ----------
    comm : MPI communicator
        MPI communicator.
    filename : str, optional
        Path to JSON file containing paths for probes, mesh and output data. Default is None.
        If probes are passed as agrument, the JSON part containing this is ignored.
        If msh is passed as argument, the JSON part containing this is ignored.
        If this file is not passed, the output data is written to the current directory.
        If this file is not passed, the msh and probes must be arguments.
    probes : ndarray, optional
        2D array of probe coordinates. shape = (n_probes, 3). Default is None.
        If this is passed, the probes are scattered to all ranks from rank 0.
        any probe object that was not passed in rank 0 will be ignored.
    msh : Mesh, optional
        If this is passed, the mesh is assigned from this object.
        The mesh object is by default a distributed data type and it is treated as so.
        In other words, the mesh is not scattered to all ranks.
    write_coords : bool
        If True, the probe coordinates are written to a csv file. Default is True.
    progress_bar : bool
        If True, a progress bar is displayed. Default is False.
    point_interpolator_type : str
        Type of point interpolator. Default is single_point_legendre.
        options are: single_point_legendre, single_point_lagrange,
        multiple_point_legendre_numpy, multiple_point_legendre_torch.
    max_pts : int, optional
        Maximum number of points to interpolate. Default is 128. Used if multiple point interpolator is selected.
    find_points_comm_pattern : str
        Communication pattern for finding points. Default is point_to_point.
        options are: point_to_point, collective.
    elem_percent_expansion : float
        Percentage expansion of the element bounding box. Default is 0.01.
    global_tree_type : str
        How is the global tree constructed to determine rank candidates for the probes.
        Only really used if using tree structures to determine candidates. Default is rank_bbox.
        options are: rank_bbox, domain_binning.
    global_tree_nbins : int
        Number of bins in the global tree. Only used if the global tree is domain_binning.
        Default is 1024.
    use_autograd : bool
        If True, autograd is used to compute the interpolation. Default is False.
        This is only used if the interpolator is a torch interpolator.

    Attributes
    ----------
    probes : ndarray
        2D array of probe coordinates. shape = (n_probes, 3). Held at Rank 0.
    interpolated_fields : ndarray
        2D array of interpolated fields at probes. shape = (n_probes, n_fields + 1). Held at Rank 0.
        The first column is always time, the rest are the interpolated fields.

    Notes
    -----
    A sample input file can be found in the examples folder of the main repository,
    However, the file is not used in said example.

    Examples
    --------
    1. Initialize from file:

    >>> from mpi4py import MPI
    >>> from pynektools.interpolation.probes import Probes
    >>> comm = MPI.COMM_WORLD
    >>> probes = Probes(comm, filename="path/to/params.json")

    2. Initialize from code, passing everything as arguments.
    Assume msh is created. One must then create the probe data in
    rank 0. A dummy probe_data must be created in all other ranks

    >>> from mpi4py import MPI
    >>> from pynektools.interpolation.probes import Probes
    >>> comm = MPI.COMM_WORLD
    >>> if comm.Get_rank() == 0:
    >>>     probes_data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    >>> else:
    >>>    probes_data = 1
    >>> probes = Probes(comm, probes=probes_data, msh=msh)

    Note that probes is initialized in all ranks, but the probe_data containing
    the coordinates are only relevant in rank 0. They are scattered internally.
    """

    class IoData:
        """
        Class to store the input/output data.

        Support class to hold paths to probe, msh, and output data.

        Parameters
        ----------
        params_file : dict
            Dictionary containing the paths to the data. Extracted from JSON input.
        """

        def __init__(self, params_file):
            self.casename = params_file["casename"]
            self.dataPath = params_file.get("dataPath", "./")
            self.index = params_file.get("first_index", 0)

    def __init__(
        self,
        comm,
        output_fname: str = "./interpolated_fields.csv",
        probes: Union[np.ndarray, str] = None,
        msh= Union[Mesh, str],
        write_coords: bool =True,
        progress_bar: bool =False,
        point_interpolator_type: str ="single_point_legendre",
        max_pts: int =128,
        find_points_comm_pattern: str ="point_to_point",
        elem_percent_expansion: float =0.01,
        global_tree_type: str ="rank_bbox",
        global_tree_nbins: int =1024,
        use_autograd: bool =False,
        find_points_tol: float =np.finfo(np.double).eps * 10,
        find_points_max_iter: int =50,
    ):

        rank = comm.Get_rank()
        self.log = Logger(comm=comm, module_name="Probes")

        self.log.tic()
        self.log.write("info", "Initializing Probes object")

        # Assign probes
        if isinstance(probes, np.ndarray) or isinstance(probes, NoneType):
            self.log.write(
                "info", "Probes provided as keyword argument"
            )
            self.probes = probes
        else: 
            print("ERROR: Probes must be provided as a string, numpy array or None if the probes are not distributed") 
            
            comm.Abort(1)
             
        # Check if the probes are distributed
        self.distributed_probes = False
        if isinstance(probes, np.ndarray):
            this_rank_has_probes_flag = 1
        else:
            this_rank_has_probes_flag = 0
    
        flag_from_all_ranks = np.zeros((comm.Get_size()), dtype=np.int64)
        flag_from_this_rank = (
            np.ones((1), dtype=np.int64) * this_rank_has_probes_flag
        )
        comm.Allgather(
            flag_from_this_rank, flag_from_all_ranks
        )

        if np.all(flag_from_all_ranks) == 1 and comm.Get_size() > 1:
            self.distributed_probes = True

        self.log.write(
            "info",
            f"Input probes are distributed: {self.distributed_probes}",
        )
        if self.distributed_probes:
            self.log.write(
                "warning",
                "Probes are distributed, all the points input to all ranks will be processed",
            )
            self.log.write(
                "warning",
                "If you are passing the same points in all ranks, you are replicating data AND interpolating the same point multiple times",
            )
            self.log.write(
                "warning",
                "If all ranks have the same points, then pass probes=None in all ranks but 0",
            )

        # Assign mesh data
        if isinstance(msh, Mesh):
            self.log.write("info", "Mesh provided as keyword argument")
            self.x = msh.x
            self.y = msh.y
            self.z = msh.z
        else:
            raise ValueError("msh must be provided as argument")

        # Initialize the interpolator
        self.log.write("info", "Initializing interpolator")
        self.itp = Interpolator(
            self.x,
            self.y,
            self.z,
            self.probes,
            comm,
            progress_bar,
            point_interpolator_type=point_interpolator_type,
            max_pts=max_pts,
            use_autograd=use_autograd,
        )

        # Set up the global tree
        self.log.write("info", "Setting up global tree")
        self.itp.set_up_global_tree(
            comm,
            find_points_comm_pattern=find_points_comm_pattern,
            global_tree_type=global_tree_type,
            global_tree_nbins=global_tree_nbins,
        )

        # Scatter the probes to all ranks
        if self.distributed_probes:
            self.log.write("info", "Assigning input probes to be a probe partition")
            self.itp.assign_local_probe_partitions()
        else:
            self.log.write("info", "Scattering probes to all ranks")
            self.itp.scatter_probes_from_io_rank(0, comm)

        # Find where the point in each rank should be
        self.log.write("info", "Finding points")
        self.itp.find_points(
            comm,
            find_points_comm_pattern=find_points_comm_pattern,
            elem_percent_expansion=elem_percent_expansion,
            tol=find_points_tol,
            max_iter=find_points_max_iter,
        )

        # Send points to the owners
        if self.distributed_probes:
            self.log.write("info", "Redistributing probes to found owners")
            self.itp.redistribute_probes_to_owners()
        else:
            # Gather probes to rank 0 again
            self.log.write("info", "Gathering probes to rank 0 after search")
            self.itp.gather_probes_to_io_rank(0, comm)

            # Redistribute the points
            self.log.write("info", "Redistributing probes to found owners")
            self.itp.redistribute_probes_to_owners_from_io_rank(0, comm)

        self.output_fname = output_fname
        if write_coords:

            # Write the coordinates to a file 
            if not self.distributed_probes:
                if comm.Get_rank() == 0:
                    write_coordinates(self, parallel=False)
            else:
                write_coordinates(self, parallel=True)

            # Write the points with warnings
            if not self.distributed_probes:
                if comm.Get_rank() == 0:
                    write_warnings(self, parallel=False)
            else:
                write_warnings(self, parallel=True)
        
        ## init dummy variables
        self.fld_data = None
        self.list_of_fields = None
        self.list_of_qoi = None
        self.number_of_files = None
        self.number_of_fields = None
        self.my_interpolated_fields = None
        self.interpolated_fields = None

        self.log.write("info", "Probes object initialized")
        self.log.toc()

    def read_fld_file(self, file_number, comm):
        """
        Method to read an fld file and return a hexadata object.

        This method wraps the preadnek function from the io.neksuite module.
        Here we use the same file name as specified in the input json.

        Parameters
        ----------
        file_number : int
            The file number to read.
            This is the number that is appended to the file name given at init.

        comm : Comm
            MPI communicator.

        Returns
        -------
        hexadata
            Hexadata object containing the field data.
        """
        self.fld_data = self.IoData(self.params_file["case"]["IO"]["fld_data"])
        self.list_of_fields = self.params_file["case"]["interpolate_fields"][
            "field_type"
        ]
        self.list_of_qoi = self.params_file["case"]["interpolate_fields"]["field"]
        self.number_of_files = self.params_file["case"]["interpolate_fields"][
            "number_of_files"
        ]
        self.number_of_fields = len(self.list_of_fields)

        # read field data
        fld_fname = (
            self.fld_data.dataPath
            + self.fld_data.casename
            + "0.f"
            + str(self.fld_data.index + file_number).zfill(5)
        )

        self.log.write("info", "Reading file: {}".format(fld_fname))
        fld_data = preadnek(fld_fname, comm)

        return fld_data

    def interpolate_from_hexadata_and_writecsv(self, fld_data, comm, mode="rst"):
        """
        Method to interpolate fields from a hexadata object and write to a csv file.

        This method interpolates the fields with key and index specified in the input json file.

        The result of the interpolation results are both held in rank 0 and also written to a csv file.

        Parameters
        ----------
        fld_data : hexadata
            Hexadata object containing the field data.

        comm : Comm
            MPI communicator.

        mode : str
            Mode of interpolation. only rst. This should be removed as an option.

        Examples
        --------
        This method can be used in conjuction to the read_fld_file method to interpolate and write to a csv file.

        >>> fld_data = probes.read_fld_file(0, comm)
        >>> probes.interpolate_from_hexadata_and_writecsv(fld_data, comm)
        """

        self.log.write("info", "Interpolating fields from hexadata object")

        self.fld_data = self.IoData(self.params_file["case"]["IO"]["fld_data"])
        self.list_of_fields = self.params_file["case"]["interpolate_fields"][
            "field_type"
        ]
        self.list_of_qoi = self.params_file["case"]["interpolate_fields"]["field"]
        self.number_of_files = self.params_file["case"]["interpolate_fields"][
            "number_of_files"
        ]
        self.number_of_fields = len(self.list_of_fields)

        # Allocate interpolated fields
        self.my_interpolated_fields = np.zeros(
            (self.itp.my_probes.shape[0], self.number_of_fields + 1), dtype=np.double
        )
        if comm.Get_rank() == 0:
            self.interpolated_fields = np.zeros(
                (self.probes.shape[0], self.number_of_fields + 1), dtype=np.double
            )
        else:
            self.interpolated_fields = None

        # Set the time
        self.my_interpolated_fields[:, 0] = fld_data.time

        for i in range(0, self.number_of_fields):
            field = get_field_from_hexadata(
                fld_data, self.list_of_fields[i], self.list_of_qoi[i]
            )

            self.log.write(
                "info",
                f"Interpolating field: {self.list_of_fields[i]}:{self.list_of_qoi[i]}",
            )
            if mode == "rst":
                self.my_interpolated_fields[:, i + 1] = (
                    self.itp.interpolate_field_from_rst(field)[:]
                )

        # Write to the csv file
        root = 0
        sendbuf = self.my_interpolated_fields.reshape(
            (self.my_interpolated_fields.size)
        )
        recvbuf, _ = self.itp.rt.gather_in_root(sendbuf, root, np.double)

        if not isinstance(recvbuf, NoneType):
            tmp = recvbuf.reshape(
                (
                    int(recvbuf.size / (self.number_of_fields + 1)),
                    self.number_of_fields + 1,
                )
            )

            # IMPORTANT - After gathering, remember to sort to the way that it
            # should be in rank zero to write values out (See the scattering
            # routine if you forget this)
            # The reason for this is that to scatter we need the data to be contigous.
            # You sort to make sure that the data from each rank is grouped.

            self.interpolated_fields[self.itp.sort_by_rank] = tmp

            # Write data in the csv file
            self.log.write(
                "info", f"Writing interpolated fields to {self.output_fname}"
            )
            write_csv(self.output_fname, self.interpolated_fields, "a")

    def interpolate_from_field_list(self, t, field_list, comm, write_data=True):
        """
        Interpolate the probes from a list of fields.

        This method interpolates from a list of fields (ndarrays of shape (nelv, lz, ly, lx)).

        Parameters
        ----------
        t : float
            Time of the field data.

        field_list : list
            List of fields to interpolate. Each field is an ndarray of shape (nelv, lz, ly, lx).

        comm : Comm
            MPI communicator.

        write_data : bool
            If True, the interpolated data is written to a csv file. Default is True.

        Examples
        --------
        This method can be used to interpolate fields from a list of fields. If you have
        previosly obtained a set of fields u,v,w as ndarrays of shape (nelv, lz, ly, lx), you can:

        >>> probes.interpolate_from_field_list(t, [u,v,w], comm)

        The results are stored in probes.interpolated_fields attribute.
        Remember: the first column of this attribute is always the time t given.
        """

        self.log.write("info", "Interpolating fields from field list")

        self.number_of_fields = len(field_list)

        # If the probes are distributed, each rank interpolate the point that it owns physically, 
        # and the sends the data to the rank were the probe was given as an input
        if self.distributed_probes:

            # Allocate buffer that keeps the interpolated fields of the points that were input in this rank
            self.interpolated_fields = np.zeros(
                (self.probes.shape[0], self.number_of_fields + 1), dtype=np.double
            )

            # Allocate buffer to keeps the interpolated fields of the points that were sent by other ranks
            my_interpolated_fields = []
            for i in range(0, len(self.itp.my_probes)):
                my_interpolated_fields.append(
                    np.zeros((self.itp.my_probes[i].shape[0], self.number_of_fields + 1), dtype=np.double)
                )
                my_interpolated_fields[i][:, 0] = t

            # Interpolate each of the fields in the list for all sources
            for i in range(0, self.number_of_fields):

                field = field_list[i]

                self.log.write("info", f"Interpolating field {i}")
                
                interpolated_fields_from_sources = self.itp.interpolate_field_from_rst(
                    field
                )[:]

                # Put this interpolated field on the right place in the buffer for each rank that sent me data
                for j in range(0, len(self.itp.my_probes)):
                    my_interpolated_fields[j][:, i + 1] = interpolated_fields_from_sources[j]

            # Send the data back to the ranks that sent me the probes
            sources, interpolated_data = self.itp.rt.all_to_all(destination=self.itp.my_sources, data= my_interpolated_fields, dtype=my_interpolated_fields[0].dtype)
            # reshape the data
            for i in range(0, len(sources)):
                interpolated_data[i] = interpolated_data[i].reshape(-1, self.number_of_fields + 1)

            
            for i in range(0, len(sources)):
                self.interpolated_fields[self.itp.local_probe_index_sent_to_destination[i]] = interpolated_data[list(sources).index(self.itp.destinations[i])][:]

        # If the probes were given in rank 0, then each rank interpolates the points that it owns physically
        # and then send them to rank 0 to be processed further 
        else:

            # Allocate interpolated fields
            self.my_interpolated_fields = np.zeros(
                (self.itp.my_probes.shape[0], self.number_of_fields + 1), dtype=np.double
            )
            if comm.Get_rank() == 0:
                self.interpolated_fields = np.zeros(
                    (self.probes.shape[0], self.number_of_fields + 1), dtype=np.double
                )
            else:
                self.interpolated_fields = None

            # Set the time
            self.my_interpolated_fields[:, 0] = t

            i = 0
            for field in field_list:

                self.log.write("info", f"Interpolating field {i}")
                self.my_interpolated_fields[:, i + 1] = self.itp.interpolate_field_from_rst(
                    field
                )[:]

                i += 1

            # Gather in rank zero for processing
            root = 0
            sendbuf = self.my_interpolated_fields.reshape(
                (self.my_interpolated_fields.size)
            )
            recvbuf, _ = self.itp.rt.gather_in_root(sendbuf, root, np.double)

            if not isinstance(recvbuf, NoneType):
                tmp = recvbuf.reshape(
                    (
                        int(recvbuf.size / (self.number_of_fields + 1)),
                        self.number_of_fields + 1,
                    )
                )
                # IMPORTANT - After gathering, remember to sort to the way that it should
                #  be in rank zero to write values out
                # (See the scattering routine if you forget this)
                # The reason for this is that to scatter we need the data to be contigous.
                # You sort to make sure that the data from each rank is grouped.
                self.interpolated_fields[self.itp.sort_by_rank] = tmp

        # Write the data
        if write_data:
            if not self.distributed_probes:
                if comm.Get_rank() == 0:
                    write_interpolated_data(self, parallel=False)
            else:
                write_interpolated_data(self, parallel=True)

def get_coordinates_from_hexadata(data):
    """Get the coordinates from a hexadata object

    Parameters
    ----------
    data :


    Returns
    -------

    """

    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    x = np.zeros((nelv, lz, ly, lx), dtype=np.double)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    for e in range(0, nelv):
        x[e, :, :, :] = data.elem[e].pos[0, :, :, :]
        y[e, :, :, :] = data.elem[e].pos[1, :, :, :]
        z[e, :, :, :] = data.elem[e].pos[2, :, :, :]

    return x, y, z

def get_field_from_hexadata(data, prefix, qoi):
    """Get field-like array from hexadata

    Parameters
    ----------
    data :

    prefix :

    qoi :


    Returns
    -------

    """
    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    field = np.zeros((nelv, lz, ly, lx), dtype=np.double)

    if prefix == "vel":
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].vel[qoi, :, :, :]

    if prefix == "pres":
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].pres[0, :, :, :]

    if prefix == "temp":
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].temp[0, :, :, :]

    if prefix == "scal":
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].scal[qoi, :, :, :]

    return field

def write_coordinates(self, parallel=False):

    # Write the coordinates
    if self.output_fname.split(".")[-1] == "csv":
        write_coordinates_csv(self, parallel)
    else:
        raise ValueError("Output file must be a csv file")

def write_interpolated_data(self, parallel=False):
    
    if self.output_fname.split(".")[-1] == "csv":
        write_interpolated_data_csv(self, parallel)
    else:
        raise ValueError("Output file must be a csv file")

def write_warnings(self, parallel=False):

    # Write out a file with the points with warnings
    indices = np.where(self.itp.err_code != 1)[0]
    # Write the points with warnings in a json file
    point_warning = {}
    # point_warning["mesh_file_path"] = msh_fld_fname
    point_warning["output_file_path"] = self.output_fname
    i = 0
    for point in indices:
        point_warning[i] = {}
        point_warning[i]["id"] = int(point)
        point_warning[i]["xyz"] = [
            float(self.itp.probes[point, 0]),
            float(self.itp.probes[point, 1]),
            float(self.itp.probes[point, 2]),
        ]
        point_warning[i]["rst"] = [
            float(self.itp.probes_rst[point, 0]),
            float(self.itp.probes_rst[point, 1]),
            float(self.itp.probes_rst[point, 2]),
        ]
        # point_warning[i]["local_el_owner"] = int(self.itp.el_owner[point])
        point_warning[i]["global_el_owner"] = int(
            self.itp.glb_el_owner[point]
        )
        point_warning[i]["error_code"] = int(self.itp.err_code[point])
        point_warning[i]["test_pattern"] = float(
            self.itp.test_pattern[point]
        )

        i += 1
    path = os.path.dirname(self.output_fname)
    fname = os.path.basename(self.output_fname)
    fname = fname.split(".")[0]

    if not parallel:
        params_file_str = json.dumps(point_warning, indent=4)
        json_output_fname = (
            f"{path}/warning_points_{fname}.json" 
        )
    else:
        json_output_fname = (
            f"{path}/warning_points_rank_{self.itp.rt.comm.Get_rank()}_{fname}.json"
        )
        params_file_str = json.dumps(point_warning, indent=4)

    self.log.write(
        "info",
        "Writing points with warnings to {}".format(json_output_fname),
    )
    with open(json_output_fname, "w") as outfile:
        outfile.write(params_file_str)

    nfound = len(np.where(self.itp.err_code == 1)[0])
    nnotfound = len(np.where(self.itp.err_code == 0)[0])
    nwarning = len(np.where(self.itp.err_code == -10)[0])
    self.log.write(
        "info",
        f"Found {nfound} points, {nnotfound} not found, {nwarning} with warnings",
    )

    if nwarning > 0:
        self.log.write(
            "warning",
            "There are points with warnings. Check the warning file to see them (error code -10)",
        )
        self.log.write(
            "warning",
            "There are points with warnings. If test pattern is small, you can trust the interpolation",
        )

    if nnotfound > 0:
        self.log.write(
            "error",
            "Some points were not found. Check the warning file to see them (error code 0)",
        )
        self.log.write(
            "error",
            "Some points were not found. The result from their interpolation will be 0",
        )

def write_coordinates_csv(self, parallel=True):

    # Write the coordinates
    ## Set up the header
    field_type_list = None
    field_index_list = None

    ## Create the header
    if isinstance(field_type_list, NoneType) or isinstance(
        field_index_list, NoneType
    ):
        header = [self.probes.shape[0], 0, 0]
    else:
        header = [self.probes.shape[0], len(field_type_list)]
        for i in range(len(field_type_list)):
            header.append(f"{field_type_list[i]}{field_index_list[i]}")

    ## Write the coordinates
    self.log.write(
        "info", "Writing probe coordinates to {}".format(self.output_fname)
    )

    if not parallel:
        write_csv(self.output_fname, self.probes, "w", header=header)
    else:
        path = os.path.dirname(self.output_fname)
        fname = os.path.basename(self.output_fname)
        write_csv(f"{path}/rank_{self.itp.rt.comm.Get_rank()}_{fname}", self.itp.probes, "w", header=header)

def write_interpolated_data_csv(self, parallel=True):

    if not parallel:

        self.log.write(
            "info", f"Writing interpolated fields to {self.output_fname}"
        )
        write_csv(self.output_fname, self.interpolated_fields, "a")

    else:
        path = os.path.dirname(self.output_fname)
        fname = os.path.basename(self.output_fname)
        write_csv(f"{path}/rank_{self.itp.rt.comm.Get_rank()}_{fname}", self.interpolated_fields, "a")



def write_csv(fname, data, mode, header=None):
    """write point positions to the file

    Parameters
    ----------
    fname :

    data :

    mode :


    Returns
    -------

    """

    # string = "writing .csv file as " + fname
    # print(string)

    # open file
    outfile = open(fname, mode)

    writer = csv.writer(outfile)

    if not isinstance(header, NoneType):
        writer.writerow(header)

    for il in range(data.shape[0]):
        data_pos = data[il, :]
        writer.writerow(data_pos)