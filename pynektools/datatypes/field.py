""" Contians class that contains information associated to fields"""

import numpy as np
from pympler import asizeof
from ..monitoring.logger import Logger

NoneType = type(None)


class Field:
    """
    Class that contains fields.

    This is the main class used to contain data that can be used for post processing.
    The data does not generarly need to be present in this class, as it is typically enough to have
    the data as ndarrays of shape (nelv, lz, ly, lx) for each field. However this class
    provides a easy interface to collect data tha is somehow associated.

    It also allows to easily write data to disk. As all the data in this class will be stored in the same file.

    Parameters
    ----------
    comm : Comm
        MPI comminicator object.
    data : HexaData, optional
        HexaData object that contains the coordinates of the domain.

    Attributes
    ----------
    fields : dict
        Dictionary that contains the fields. The keys are the field names and the values are lists of ndarrays.
        The keys for these dictionaries are the same as for Hexadata objects, i.e. vel, pres, temp, scal.
    vel_fields : int
        Number of velocity fields.
    pres_fields : int
        Number of pressure fields.
    temp_fields : int
        Number of temperature fields.
    scal_fields : int
        Number of scalar fields.
    t : float
        Time of the data.

    Returns
    -------

    Examples
    --------
    If a hexadata object: data is read from disk, the field object can be created directly from it.

    >>> from pynektools.datatypes.field import Field
    >>> fld = Mesh(comm, data = data)

    If one wishes to use the data in the fields. It is possible to reference it with a ndarray of shape (nelv, lz, ly, lx)
    as follows:

    >>> u = fld.fields["vel"][0]
    >>> v = fld.fields["vel"][1]
    >>> w = fld.fields["vel"][2]
    >>> vel_magnitude = np.sqrt(u**2 + v**2 + w**2)

    A field object can be created empty and then fields can be added to it. Useful to write data to disk.
    if a ndarray u is created with shape (nelv, lz, ly, lx) it can be added to the field object as follows:

    >>> from pynektools.datatypes.field import Field
    >>> fld = Field(comm)
    >>> fld.fields["vel"].append(u)
    >>> fld.update_vars()

    This fld object can then be used to write fld files from field u created in the code.
    """

    def __init__(self, comm, data=None):

        self.log = Logger(comm=comm, module_name="Field")

        self.fields = {}
        self.fields["vel"] = []
        self.fields["pres"] = []
        self.fields["temp"] = []
        self.fields["scal"] = []
        self.t = 0.0
        self.vel_fields = 0
        self.pres_fields = 0
        self.temp_fields = 0
        self.scal_fields = 0

        if not isinstance(data, NoneType):

            self.log.tic()
            self.log.write("info", "Initializing Field object from HexaData")

            vars_ = data.var
            self.vel_fields = vars_[1]
            self.pres_fields = vars_[2]
            self.temp_fields = vars_[3]
            self.scal_fields = vars_[4]

            # Read the full fields
            for qoi in range(0, self.vel_fields):
                prefix = "vel"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.pres_fields):
                prefix = "pres"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.temp_fields):
                prefix = "temp"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.scal_fields):
                prefix = "scal"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            self.t = data.time

            self.log.write("info", "Field object initialized")
            self.log.toc()
        else:
            self.log.write("info", "Initializing empty Field object")

    def __memory_usage__(self, comm):
        """
        Print the memory usage of the object.

        This function is used to print the memory usage of the object.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.

        Returns
        -------
        None

        """

        memory_usage = asizeof.asizeof(self) / (1024**2)  # Convert bytes to MB
        print(f"Rank: {comm.Get_rank()} - Memory usage of Field: {memory_usage} MB")

    def update_vars(self):
        """
        Update number of fields.

        Update the number of fields in the class in the event that
        it has been modified. This is needed for writing data properly if more arrays are added to the class.

        Examples
        --------
        A field object can be created empty and then fields can be added to it. Useful to write data to disk.
        if a ndarray u is created with shape (nelv, lz, ly, lx) it can be added to the field object as follows:

        >>> from pynektools.datatypes.field import Field
        >>> fld = Field(comm)
        >>> fld.fields["vel"].append(u)
        >>> fld.update_vars()

        This fld object can then be used to write fld files from field u created in the code.
        """
        self.vel_fields = len(self.fields["vel"])
        self.pres_fields = len(self.fields["pres"])
        self.temp_fields = len(self.fields["temp"])
        self.scal_fields = len(self.fields["scal"])

        self.log.write("info", "Field variables updated")
        self.log.write(
            "info",
            f"Velocity fields: {self.vel_fields}, Pressure fields: {self.pres_fields}, Temperature fields: {self.temp_fields}, Scalar fields: {self.scal_fields}",
        )


def get_field_from_hexadata(data, prefix, qoi):
    """
    Extract a field from the hexadata object and return it as a numpy array

    This way the hexadata can be more readily used for computations.

    Parameters
    ----------
    data : hexadata
        The hexadata object that contains the field data.
    prefix : str
        The prefix of the field to extract. Options are "vel", "pres", "temp", "scal"
    qoi : int
        The quantity of interest to extract, e.g. if prefix is "vel" and qoi is 0, the velocity field in x durection will be extracted.

    Returns
    -------
    ndarray
        The field data extracted from the hexadata object
    """
    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    if prefix == "vel":
        field = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].vel.dtype)
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].vel[qoi, :, :, :]

    if prefix == "pres":
        field = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].pres.dtype)
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].pres[0, :, :, :]

    if prefix == "temp":
        field = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].temp.dtype)
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].temp[0, :, :, :]

    if prefix == "scal":
        field = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].scal.dtype)
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].scal[qoi, :, :, :]

    return field
