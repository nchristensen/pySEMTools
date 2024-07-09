""" Contians class that contains information associated to fields"""

import numpy as np

NoneType = type(None)


class Field:
    """Class that contains information associated to fields"""

    def __init__(self, comm, data=None):

        self.fields = {}
        self.fields["vel"] = []
        self.fields["pres"] = []
        self.fields["temp"] = []
        self.fields["scal"] = []
        if not isinstance(data, NoneType):
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

    def update_vars(self):
        """Update the number of fields in the class in the event that
        it has been modified. This is needed for writing data properly"""
        self.vel_fields = len(self.fields["vel"])
        self.pres_fields = len(self.fields["pres"])
        self.temp_fields = len(self.fields["temp"])
        self.scal_fields = len(self.fields["scal"])


def get_field_from_hexadata(data, prefix, qoi):
    """Extract a field from the hexadata object and return it as a numpy array"""
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
