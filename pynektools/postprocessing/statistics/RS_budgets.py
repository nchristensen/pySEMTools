###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# generic function to compute the gradient of a scalar field
###########################################################################################
###########################################################################################
def compute_scalar_first_derivative(comm, msh, coef, scalar, scalar_deriv):
    if msh.gdim == 3:
        scalar_deriv.c1 = coef.dudxyz(scalar, coef.drdx, coef.dsdx, coef.dtdx)
        scalar_deriv.c2 = coef.dudxyz(scalar, coef.drdy, coef.dsdy, coef.dtdy)
        scalar_deriv.c3 = coef.dudxyz(scalar, coef.drdz, coef.dsdz, coef.dtdz)
    elif msh.gdim == 2:
        scalar_deriv.c1 = coef.dudxyz(scalar, coef.drdx, coef.dsdx, coef.dtdx)
        scalar_deriv.c2 = coef.dudxyz(scalar, coef.drdy, coef.dsdy, coef.dtdy)
        scalar_deriv.c3 = 0.0 * scalar_deriv.c2
    else:
        import sys

        sys.exit("supports either 2D or 3D data")


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# do dssum on a vector with components c1, c2, c3
###########################################################################################
###########################################################################################
def do_dssum_on_3comp_vector(dU_dxi, coef, msh):
    coef.dssum(dU_dxi.c1, msh)
    coef.dssum(dU_dxi.c2, msh)
    coef.dssum(dU_dxi.c3, msh)


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# generic function to compute the diagonal second derivatives of a scalar field from its gradient
###########################################################################################
###########################################################################################
def compute_scalar_second_derivative(comm, msh, coef, scalar_deriv, scalar_deriv2):
    if msh.gdim == 3:
        scalar_deriv2.c1 = coef.dudxyz(scalar_deriv.c1, coef.drdx, coef.dsdx, coef.dtdx)
        scalar_deriv2.c2 = coef.dudxyz(scalar_deriv.c2, coef.drdy, coef.dsdy, coef.dtdy)
        scalar_deriv2.c3 = coef.dudxyz(scalar_deriv.c3, coef.drdz, coef.dsdz, coef.dtdz)
    elif msh.gdim == 2:
        scalar_deriv2.c1 = coef.dudxyz(scalar_deriv.c1, coef.drdx, coef.dsdx, coef.dtdx)
        scalar_deriv2.c2 = coef.dudxyz(scalar_deriv.c2, coef.drdy, coef.dsdy, coef.dtdy)
        scalar_deriv2.c3 = 0.0 * scalar_deriv2.c3
    else:
        import sys

        sys.exit("supports either 2D or 3D data")


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# generic function to write a 9 component field with input as 3 vectors of 3 components each
###########################################################################################
###########################################################################################
def write_file_9c(comm, msh, dU_dxi, dV_dxi, dW_dxi, fname_gradU, if_write_mesh):
    from pynektools.datatypes.field import FieldRegistry
    from pynektools.io.ppymech.neksuite import pynekwrite
    import numpy as np

    gradU = FieldRegistry(comm)

    gradU.add_field(comm, field_name="c1", field=dU_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c2", field=dU_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c3", field=dU_dxi.c3, dtype=np.single)
    gradU.add_field(comm, field_name="c4", field=dV_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c5", field=dV_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c6", field=dV_dxi.c3, dtype=np.single)
    gradU.add_field(comm, field_name="c7", field=dW_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c8", field=dW_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c9", field=dW_dxi.c3, dtype=np.single)

    pynekwrite(fname_gradU, comm, msh=msh, fld=gradU, wdsz=4, write_mesh=if_write_mesh)

    gradU.clear()


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# generic function to write a 6-component field with input as 2 vectors of 3 components each
###########################################################################################
###########################################################################################
def write_file_6c(comm, msh, dU_dxi, dV_dxi, fname_gradU, if_write_mesh):
    from pynektools.datatypes.field import FieldRegistry
    from pynektools.io.ppymech.neksuite import pynekwrite
    import numpy as np

    gradU = FieldRegistry(comm)

    gradU.add_field(comm, field_name="c1", field=dU_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c2", field=dU_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c3", field=dU_dxi.c3, dtype=np.single)
    gradU.add_field(comm, field_name="c4", field=dV_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c5", field=dV_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c6", field=dV_dxi.c3, dtype=np.single)

    pynekwrite(fname_gradU, comm, msh=msh, fld=gradU, wdsz=4, write_mesh=if_write_mesh)

    gradU.clear()


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# generic function to write a 3-component (vector) field
###########################################################################################
###########################################################################################
def write_file_3c(comm, msh, dU_dxi, fname_gradU, if_write_mesh):
    from pynektools.datatypes.field import FieldRegistry
    from pynektools.io.ppymech.neksuite import pynekwrite
    import numpy as np

    gradU = FieldRegistry(comm)

    gradU.add_field(comm, field_name="c1", field=dU_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c2", field=dU_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c3", field=dU_dxi.c3, dtype=np.single)

    pynekwrite(fname_gradU, comm, msh=msh, fld=gradU, wdsz=4, write_mesh=if_write_mesh)

    gradU.clear()


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
## generic function to give the file name depending on the code used
###########################################################################################
###########################################################################################
# function to give the file name based on code, etc.
def give_me_the_stat_file_name(
    which_dir, fname_base, stat_file_number, which_code="NEKO", nek5000_stat_type="s"
):
    if which_code.casefold() == "neko":
        output_file_name = which_dir + "/" + fname_base
    elif which_code.casefold() == "nek5000":
        output_file_name = (
            which_dir + "/" + nek5000_stat_type + stat_file_number + fname_base
        )
    # print('filename here was: ', output_file_name )
    return output_file_name


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
## genertic function to compute the additional fields required for budget terms, etc.
###########################################################################################
###########################################################################################
def compute_and_write_additional_pstat_fields(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    if_write_mesh=False,
    which_code="NEKO",
    nek5000_stat_type="s",
    if_do_dssum_on_derivatives=False,
):

    ###########################################################################################
    # do some initial checks
    import sys

    # see if nek5000 file names, etc. are correct
    if which_code.casefold() == "nek5000":
        if fname_mean != fname_stat:
            sys.exit(
                "for NEK5000 statistics fname_mean must be the same as fname_stat and equal to casename0.fXXXXX"
            )
        if nek5000_stat_type != "s" and nek5000_stat_type != "t":
            sys.exit(
                'for NEK5000 statistics nek5000_stat_type can be either "s" or "t"'
            )

    ###########################################################################################
    import warnings
    from mpi4py import MPI  # equivalent to the use of MPI_init() in C
    import numpy as np

    from pynektools.datatypes.msh import Mesh
    from pynektools.datatypes.field import FieldRegistry
    from pynektools.datatypes.coef import Coef
    from pynektools.io.ppymech.neksuite import pynekread

    ###########################################################################################
    # Get mpi info
    comm = MPI.COMM_WORLD

    ###########################################################################################
    # intialize the mesh and some fields
    if if_do_dssum_on_derivatives:
        msh = Mesh(comm, create_connectivity=True)
    else:
        msh = Mesh(comm, create_connectivity=False)

    mean_fields = FieldRegistry(comm)
    stat_fields = FieldRegistry(comm)

    dU_dxi = FieldRegistry(comm)
    dV_dxi = FieldRegistry(comm)
    dW_dxi = FieldRegistry(comm)

    # pressure gradient and scond derivative
    dP_dxi = FieldRegistry(comm)
    d2P_dxi2 = FieldRegistry(comm)

    # pressure tranposrt
    dPU_dxi = FieldRegistry(comm)
    dPV_dxi = FieldRegistry(comm)
    dPW_dxi = FieldRegistry(comm)

    # generic quantity
    dQ_dxi = FieldRegistry(comm)
    d2Q_dxi2 = FieldRegistry(comm)

    ###########################################################################################
    # using the same .fXXXXXX extenstion as the mean fields
    this_ext = fname_mean[-8:]
    this_ext_check = fname_stat[-8:]

    # check the two files match. can be replaced by an error, but that seems too harsh and limiting
    if this_ext != this_ext_check:
        warnings.warn(
            "File index of fname_stat and fname_mean differ! Hope you know what you are doing!"
        )

    fname_gradU = which_dir + "/dUdx" + this_ext
    fname_hessU = which_dir + "/d2Udx2" + this_ext
    fname_derivP = which_dir + "/dnPdxn" + this_ext
    fname_gradPU = which_dir + "/dPUdx" + this_ext
    # full_fname_mean = which_dir+"/"+fname_mean
    # full_fname_stat = which_dir+"/"+fname_stat
    full_fname_mesh = which_dir + "/" + fname_mesh

    ###########################################################################################
    # read mesh and compute coefs
    pynekread(full_fname_mesh, comm, msh=msh, data_dtype=np.single)

    if msh.gdim < 3:
        sys.exit("only 3D data is supported at the moment")

    coef = Coef(msh, comm, get_area=False)

    ###########################################################################################
    # define file_keys of the fields based on the codes

    # direty fix: not needed for neko. but can't leave it empty or undefined
    stat_file_number_PU = ["03", "04", "04"]
    stat_file_number_UiUj = ["02", "02", "02", "03", "03", "03"]
    stat_file_number_UiUjUk = [
        "06",
        "07",
        "07",
        "07",
        "07",
        "08",
        "09",
        "08",
        "08",
        "08",
    ]

    if which_code.casefold() == "neko":
        file_keys_mean_fields = ["vel_0", "vel_1", "vel_2", "pres"]

        # key names taken from: https://neko.cfd/docs/develop/df/d8f/statistics-guide.html
        #                            "PU"       "PV"      "PW"
        file_keys_PU = ["scal_17", "scal_18", "scal_19"]

        #                           "UU"  , "VV"  , "WW"  , "UV" ,  "UW"  ,  "VW"  ]
        file_keys_UiUj = ["vel_0", "vel_1", "vel_2", "temp", "scal_0", "scal_1"]

        #                           "UUU"  , "VVV"  , "WWW"  , "UUV"  , "UUW"  , "UVV"  , "UVW"  , "VVW"  ,  "UWW"  ,  "VWW"
        file_keys_UiUjUk = [
            "scal_2",
            "scal_3",
            "scal_4",
            "scal_5",
            "scal_6",
            "scal_7",
            "scal_8",
            "scal_9",
            "scal_10",
            "scal_11",
        ]

    elif which_code.casefold() == "nek5000":
        file_keys_mean_fields = ["vel_0", "vel_1", "vel_2", "temp"]

        # key names taken from https://github.com/KTH-Nek5000/KTH_Toolbox/blob/devel/tools/stat/stat_IO.f
        #                       "PU"    "PV"    "PW"
        file_keys_PU = ["temp", "vel_0", "vel_1"]

        #                           "UU"  , "VV"  , "WW"  , "UV"  , "UW"  , "VW"
        file_keys_UiUj = [
            "vel_0",
            "vel_1",
            "vel_2",
            "vel_0",
            "vel_2",
            "vel_1",
        ]  # not a mistake! nek5000's order is a bit strange

        #                          "UUU" , "VVV" , "WWW" , "UUV" , "UUW", "UVV" , "UVW" , "VVW" , "UWW" ,"VWW"
        file_keys_UiUjUk = [
            "temp",
            "vel_0",
            "vel_1",
            "vel_2",
            "temp",
            "vel_0",
            "vel_2",
            "vel_1",
            "vel_2",
            "temp",
        ]

    else:
        sys.exit("which_code can be either NEKO or NEK5000")

    ###########################################################################################
    # Read velocity and pressure
    # if which_code.casefold()=="neko":
    #     this_file_name = full_fname_mean
    # elif which_code.casefold()=="nek5000":
    #     this_file_name = which_dir+"/"+nek5000_stat_type+"01"+fname_mean
    this_file_name = give_me_the_stat_file_name(
        which_dir, fname_mean, "01", which_code, nek5000_stat_type
    )
    mean_fields.add_field(
        comm,
        field_name="U",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_mean_fields[0],
        dtype=np.single,
    )
    mean_fields.add_field(
        comm,
        field_name="V",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_mean_fields[1],
        dtype=np.single,
    )
    mean_fields.add_field(
        comm,
        field_name="W",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_mean_fields[2],
        dtype=np.single,
    )
    mean_fields.add_field(
        comm,
        field_name="P",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_mean_fields[3],
        dtype=np.single,
    )

    ###########################################################################################
    # velocity first and second derivatives
    ###########################################################################################
    #
    compute_scalar_first_derivative(comm, msh, coef, mean_fields.registry["U"], dU_dxi)
    compute_scalar_first_derivative(comm, msh, coef, mean_fields.registry["V"], dV_dxi)
    compute_scalar_first_derivative(comm, msh, coef, mean_fields.registry["W"], dW_dxi)
    if (
        if_do_dssum_on_derivatives
    ):  # this could be a terrible idea to do dssum on a vector that is then differentiated again
        if comm.Get_rank() == 0:
            warnings.warn(
                "you are doing dssum on a derivative that you are differentiating again. maybe change this."
            )
        do_dssum_on_3comp_vector(dU_dxi, coef, msh)
        do_dssum_on_3comp_vector(dV_dxi, coef, msh)
        do_dssum_on_3comp_vector(dW_dxi, coef, msh)
    write_file_9c(
        comm, msh, dU_dxi, dV_dxi, dW_dxi, fname_gradU, if_write_mesh=if_write_mesh
    )

    compute_scalar_second_derivative(comm, msh, coef, dU_dxi, dU_dxi)
    compute_scalar_second_derivative(comm, msh, coef, dV_dxi, dV_dxi)
    compute_scalar_second_derivative(comm, msh, coef, dW_dxi, dW_dxi)
    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dU_dxi, coef, msh)
        do_dssum_on_3comp_vector(dV_dxi, coef, msh)
        do_dssum_on_3comp_vector(dW_dxi, coef, msh)
    write_file_9c(
        comm, msh, dU_dxi, dV_dxi, dW_dxi, fname_hessU, if_write_mesh=if_write_mesh
    )

    ###############################
    # free up some memory
    ###############################
    del dU_dxi
    del dV_dxi
    del dW_dxi
    # dU_dxi.clear()
    # dV_dxi.clear()
    # dW_dxi.clear()
    # d2U_dxi2.clear()
    # d2V_dxi2.clear()
    # d2W_dxi2.clear()
    ################################

    ###########################################################################################
    # pressure first and second derivatives
    ###########################################################################################
    compute_scalar_first_derivative(comm, msh, coef, mean_fields.registry["P"], dP_dxi)
    compute_scalar_second_derivative(comm, msh, coef, dP_dxi, d2P_dxi2)
    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dP_dxi, coef, msh)
        do_dssum_on_3comp_vector(d2P_dxi2, coef, msh)
    write_file_6c(
        comm, msh, dP_dxi, d2P_dxi2, fname_derivP, if_write_mesh=if_write_mesh
    )

    ###############################
    # free up some memory
    ###############################
    del dP_dxi
    del d2P_dxi2
    # dP_dxi.clear()
    # d2P_dxi2.clear()
    mean_fields.clear()  # no longer needed
    ###############################

    ###########################################################################################
    # pressure velocity product
    ###########################################################################################
    this_file_name = give_me_the_stat_file_name(
        which_dir, fname_stat, stat_file_number_PU[0], which_code, nek5000_stat_type
    )
    stat_fields.add_field(
        comm,
        field_name="PU",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_PU[0],
        dtype=np.single,
    )
    this_file_name = give_me_the_stat_file_name(
        which_dir, fname_stat, stat_file_number_PU[1], which_code, nek5000_stat_type
    )
    stat_fields.add_field(
        comm,
        field_name="PV",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_PU[1],
        dtype=np.single,
    )
    this_file_name = give_me_the_stat_file_name(
        which_dir, fname_stat, stat_file_number_PU[2], which_code, nek5000_stat_type
    )
    stat_fields.add_field(
        comm,
        field_name="PW",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_PU[2],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["PU"], dPU_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["PV"], dPV_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["PW"], dPW_dxi
    )
    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dPU_dxi, coef, msh)
        do_dssum_on_3comp_vector(dPV_dxi, coef, msh)
        do_dssum_on_3comp_vector(dPW_dxi, coef, msh)
    write_file_9c(
        comm, msh, dPU_dxi, dPV_dxi, dPW_dxi, fname_gradPU, if_write_mesh=if_write_mesh
    )

    stat_fields.clear()

    ###########################################################################################
    # velocity product terms
    ###########################################################################################
    # these need both first and second derivatives
    # we will be more agreesive from here on to save memory
    actual_field_names = ["UU", "VV", "WW", "UV", "UW", "VW"]

    for icomp in range(0, 6):
        if comm.Get_rank() == 0:
            print("working on: " + actual_field_names[icomp])

        # if which_code.casefold()=="neko":
        #     this_file_name=full_fname_stat
        # elif which_code.casefold()=="nek5000":
        #     this_file_name=
        this_file_name = give_me_the_stat_file_name(
            which_dir,
            fname_stat,
            stat_file_number_UiUj[icomp],
            which_code,
            nek5000_stat_type,
        )
        stat_fields.add_field(
            comm,
            field_name=actual_field_names[icomp],
            file_type="fld",
            file_name=this_file_name,
            file_key=file_keys_UiUj[icomp],
            dtype=np.single,
        )

        compute_scalar_first_derivative(
            comm, msh, coef, stat_fields.registry[actual_field_names[icomp]], dQ_dxi
        )

        ###############################
        stat_fields.clear()  # no longer needed
        ###############################

        compute_scalar_second_derivative(comm, msh, coef, dQ_dxi, d2Q_dxi2)

        if if_do_dssum_on_derivatives:
            do_dssum_on_3comp_vector(dQ_dxi, coef, msh)
            do_dssum_on_3comp_vector(d2Q_dxi2, coef, msh)

        this_file_name = (
            which_dir + "/dn" + actual_field_names[icomp] + "dxn" + this_ext
        )

        write_file_6c(
            comm, msh, dQ_dxi, d2Q_dxi2, this_file_name, if_write_mesh=if_write_mesh
        )

        ###############################
        # del d2Q_dxi2
        # d2Q_dxi2.clear()    # to not exceed a 6-component array worth of memory in this loop
        # in addition d2Q_dxi2 is no longer needed after this loop
        ###############################

    del d2Q_dxi2

    ###########################################################################################
    # tripple product terms
    ###########################################################################################
    actual_field_names = [
        "UUU",
        "VVV",
        "WWW",
        "UUV",
        "UUW",
        "UVV",
        "UVW",
        "VVW",
        "UWW",
        "VWW",
    ]

    for icomp in range(0, 10):
        if comm.Get_rank() == 0:
            print("working on: " + actual_field_names[icomp])

        this_file_name = give_me_the_stat_file_name(
            which_dir,
            fname_stat,
            stat_file_number_UiUjUk[icomp],
            which_code,
            nek5000_stat_type,
        )

        stat_fields.add_field(
            comm,
            field_name=actual_field_names[icomp],
            file_type="fld",
            file_name=this_file_name,
            file_key=file_keys_UiUjUk[icomp],
            dtype=np.single,
        )

        compute_scalar_first_derivative(
            comm, msh, coef, stat_fields.registry[actual_field_names[icomp]], dQ_dxi
        )

        if if_do_dssum_on_derivatives:
            do_dssum_on_3comp_vector(dQ_dxi, coef, msh)

        ###############################
        stat_fields.clear()  # no longer needed
        ###############################

        this_file_name = which_dir + "/d" + actual_field_names[icomp] + "dx" + this_ext

        write_file_3c(comm, msh, dQ_dxi, this_file_name, if_write_mesh=if_write_mesh)

    ###########################################################################################
    # finishing up
    ###########################################################################################
    del dQ_dxi
    # dQ_dxi.clear()
    stat_fields.clear()

    print("-------As a great man once said: run successful: dying ...")


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# user specified function to define the interpolation points
###########################################################################################
###########################################################################################
def user_defined_interpolating_points():
    import numpy as np
    import pynektools.interpolation.pointclouds as pcs
    import pynektools.interpolation.utils as interp_utils

    # from mpi4py import MPI

    # # Get mpi info
    # comm = MPI.COMM_WORLD

    # Create the coordinates of the plane you want
    x_bbox = [0.5, 2]
    y_bbox = [-1, 1]
    z_bbox = [0, 0.6]  # See how here I am just setting this to be one value

    nx = 100
    ny = 100
    nz = 7  # I want my plane to be  in z

    print("generate interpolation points")

    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")
    x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

    xyz = interp_utils.transform_from_array_to_list(nx, ny, nz, [x, y, z])

    return xyz


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# interpolate the 44+N fields onto the user specified set of points
###########################################################################################
###########################################################################################
def interpolate_all_stat_and_pstat_fields_onto_points(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    xyz,
    which_code="NEKO",
    nek5000_stat_type="s",
    if_do_dssum_before_interp=True,
    if_create_boundingBox_for_interp=False,
):

    from mpi4py import MPI  # equivalent to the use of MPI_init() in C
    import numpy as np
    from scipy.io import savemat

    from pynektools.datatypes.msh import Mesh
    from pynektools.datatypes.field import FieldRegistry
    from pynektools.datatypes.coef import Coef
    from pynektools.io.ppymech.neksuite import pynekread
    from pymech.neksuite.field import read_header
    from pynektools.interpolation.probes import Probes

    if if_create_boundingBox_for_interp:
        from pynektools.datatypes.msh_partitioning import MeshPartitioner

    ###########################################################################################
    # do some initial checks
    import sys

    # see if nek5000 file names, etc. are correct
    if which_code.casefold() == "nek5000":
        if fname_mean != fname_stat:
            sys.exit(
                "for NEK5000 statistics fname_mean must be the same as fname_stat and equal to casename0.fXXXXX"
            )
        if nek5000_stat_type != "s" and nek5000_stat_type != "t":
            sys.exit(
                'for NEK5000 statistics nek5000_stat_type can be either "s" or "t"'
            )

    ###########################################################################################
    # Get mpi info
    comm = MPI.COMM_WORLD

    ###########################################################################################
    # intialize the mesh and some fields
    if if_create_boundingBox_for_interp:
        msh = Mesh(comm, create_connectivity=False)
    else:
        msh = Mesh(comm, create_connectivity=True)
    mean_fields = FieldRegistry(comm)

    ###########################################################################################
    # using the same .fXXXXXX extenstion as the mean fields
    this_ext = fname_mean[-8:]
    this_ext_check = fname_stat[-8:]

    # check the two files match. can be replaced by an error, but that seems too harsh and limiting
    if this_ext != this_ext_check:
        warnings.warn(
            "File index of fname_stat and fname_mean differ! Hope you know what you are doing!"
        )

    fname_gradU = which_dir + "/dUdx" + this_ext
    fname_hessU = which_dir + "/d2Udx2" + this_ext
    fname_derivP = which_dir + "/dnPdxn" + this_ext
    fname_gradPU = which_dir + "/dPUdx" + this_ext
    # full_fname_mean = which_dir+"/"+fname_mean
    # full_fname_stat = which_dir+"/"+fname_stat
    full_fname_mesh = which_dir + "/" + fname_mesh

    ###########################################################################################
    # get the file name for the 44 fileds collected in run time
    if which_code.casefold() == "neko":
        these_names = [
            give_me_the_stat_file_name(
                which_dir, fname_mean, "00", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_stat, "00", which_code, nek5000_stat_type
            ),
        ]
    elif which_code.casefold() == "nek5000":
        these_names = [
            give_me_the_stat_file_name(
                which_dir, fname_mean, "01", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "02", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "03", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "04", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "05", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "06", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "07", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "08", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "09", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "10", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "11", which_code, nek5000_stat_type
            ),
        ]

    # add the name of the additional fields, these are common between neko and nek5000
    these_names.extend([fname_gradU, fname_hessU, fname_derivP, fname_gradPU])

    actual_field_names = ["UU", "VV", "WW", "UV", "UW", "VW"]
    for icomp in range(0, 6):
        this_file_name = (
            which_dir + "/dn" + actual_field_names[icomp] + "dxn" + this_ext
        )
        these_names.append(this_file_name)

    actual_field_names = [
        "UUU",
        "VVV",
        "WWW",
        "UUV",
        "UUW",
        "UVV",
        "UVW",
        "VVW",
        "UWW",
        "VWW",
    ]
    for icomp in range(0, 10):
        this_file_name = which_dir + "/d" + actual_field_names[icomp] + "dx" + this_ext
        these_names.append(this_file_name)

    ###########################################################################################
    # read mesh and redefine it based on the boundaring box if said
    pynekread(full_fname_mesh, comm, msh=msh, data_dtype=np.single)

    if msh.gdim < 3:
        sys.exit("only 3D data is supported at the moment")

    if if_create_boundingBox_for_interp:
        xyz_max = np.max(xyz, axis=0)
        xyz_min = np.min(xyz, axis=0)

        if comm.Get_rank() == 0:
            print("xyz_min: ", xyz_min)
            print("xyz_max: ", xyz_max)

        cond = (
            (msh.x >= xyz_min[0])
            & (msh.x <= xyz_max[0])
            & (msh.y >= xyz_min[1])
            & (msh.y <= xyz_max[1])
            & (msh.z >= xyz_min[2])
            & (msh.z <= xyz_max[2])
        )

        mp = MeshPartitioner(comm, msh=msh, conditions=[cond])
        msh = mp.create_partitioned_mesh(
            msh, partitioning_algorithm="load_balanced_linear", create_conectivity=True
        )

    ###########################################################################################
    # compute coef, for interpolation
    coef = Coef(msh, comm, get_area=False)

    ###########################################################################################
    # initiate probes
    # probes = Probes(comm, probes=xyz, msh=msh, \
    #                 point_interpolator_type="multiple_point_legendre_numpy", \
    #                 global_tree_type="domain_binning" , \
    #                 max_pts = 256 )
    probes = Probes(
        comm,
        probes=xyz,
        msh=msh,
        point_interpolator_type="multiple_point_legendre_numpy",
        max_pts=128,
    )

    ###########################################################################################
    #
    # interp_fields_size = ( xyz.len(), 144 )     # enough for now
    # interp_fields = np.zeros( interp_fields_size, dtype=float, order='C', *, like=None)#

    ###########################################################################################
    for fname in these_names:

        if comm.Get_rank() == 0:
            print("----------- working on file: ", fname)

        header = read_header(fname)
        # num_fields = header.nb_vars()
        # field_names = [] # set to empty for now

        #########################
        ## THIS NEEDS TO BE FIXED
        field_names = ["vel_0", "vel_1", "vel_2", "temp"]
        #########################

        for icomp in range(0, len(field_names)):
            if comm.Get_rank() == 0:
                print(
                    "---working on field ", icomp, "from a total of ", len(field_names)
                )

            # load the field
            mean_fields.add_field(
                comm,
                field_name="tmpF",
                file_type="fld",
                file_name=fname,
                file_key=field_names[icomp],
                dtype=np.single,
            )

            if if_create_boundingBox_for_interp:
                mean_fields = mp.create_partitioned_field(
                    mean_fields, partitioning_algorithm="load_balanced_linear"
                )

            # do dssum to make it continuous
            if if_do_dssum_before_interp:
                coef.dssum(mean_fields.registry["tmpF"], msh)

            # interpolate the fields
            probes.interpolate_from_field_list(
                0, [mean_fields.registry["tmpF"]], comm, write_data=True
            )

            mean_fields.clear()


###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# the presumed workflow
###########################################################################################
###########################################################################################
# filenames
fname_mesh = "s01FD_3d0.f00001"
fname_mean = "FD_3d0.f00001"
fname_stat = "FD_3d0.f00001"
which_dir = "./"
which_code = "nek5000"
nek5000_stat_type = "s"

if_do_dssum_before_interp = False  # whether to do dssum before interpolation
if_create_boundingBox_for_interp = True
if_do_dssum_on_derivatives = True
# step 1:
# the script to average the stat files in TIME goes here

# step 1.5:
# can take the average in space here: currently only if the output is still 3D
# the filenames might need to be changed after this call
# some_function_to_average in space

# # step 2:
# # compute the additional fields based on the 44 stat fields.
# compute_and_write_additional_pstat_fields(which_dir,fname_mesh,fname_mean,fname_stat,\
#                                           if_write_mesh=True,which_code=which_code,nek5000_stat_type=nek5000_stat_type, \
#                                           if_do_dssum_on_derivatives=if_do_dssum_on_derivatives)

# step <3:
# define interpolation points
# this should be called from rank 0 only!!
xyz = user_defined_interpolating_points()

# step 3:
# interpolate the 44+N fields onto the interpolation points
# this function is not working at the moment
# For starters, we want this function to write the interpolated fields in MATLAB format so we can debug
interpolate_all_stat_and_pstat_fields_onto_points(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    xyz,
    which_code=which_code,
    nek5000_stat_type=nek5000_stat_type,
    if_do_dssum_before_interp=if_do_dssum_before_interp,
    if_create_boundingBox_for_interp=if_create_boundingBox_for_interp,
)

# step 3.5:
# potential script to perform averaging on the interpolated fields is called here

# step 4:
# function to compute budgets is called  here

# step 4.5:
# scrtipt to plot, etc. the budget terms is called here

# what is missing??

###########################################################################################
###########################################################################################
