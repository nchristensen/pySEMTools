""" Module that contains the class and methods to perform direct sampling on a field """

from mpi4py import MPI
from ..monitoring.logger import Logger
from ..datatypes.msh import Mesh
from ..datatypes.coef import Coef
from ..datatypes.coef import get_transform_matrix
import numpy as np
import bz2
import sys
import h5py
import os
import torch
import math

class DirectSampler:

    """ 
    Class to perform direct sampling on a field in the SEM format
    """

    def __init__(self, comm: MPI.Comm = None, dtype: np.dtype = np.double,  msh: Mesh = None, filename: str = None, max_elements_to_process: int = 256, bckend: str = "numpy"):
        
        self.log = Logger(comm=comm, module_name="DirectSampler")
        
        if msh is not None:
            self.init_from_msh(msh, dtype=dtype, max_elements_to_process=max_elements_to_process)
        elif filename is not None:
            self.init_from_file(comm, filename, max_elements_to_process=max_elements_to_process)
        else:
            self.log.write("info", "No mesh provided. Please provide a mesh to initialize the DirectSampler")

        # Init bckend
        self.bckend = bckend
        if bckend == "torch": 
            # Find the device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Set the device dtype
            if dtype == np.float32:
                self.dtype_d = torch.float32
            elif dtype == np.float64:
                self.dtype_d = torch.float64
            # Transfer needed data
            self.v_d = torch.tensor(self.v, dtype=self.dtype_d, device = self.device, requires_grad=False)
            self.vinv_d = torch.tensor(self.vinv, dtype=self.dtype_d, device = self.device, requires_grad=False)

    def init_from_file(self, comm: MPI.Comm, filename: str, max_elements_to_process: int = 256):
        """
        """

        self.log.write("info", f"Initializing the DirectSampler from file: {filename}")

        self.settings, self.compressed_data = self.read_compressed_samples(comm = comm, filename=filename)

        self.init_common(max_elements_to_process)

        self.uncompressed_data = self.decompress_samples(self.settings, self.compressed_data)

        self.kw_diag = self.settings["covariance"]["kw_diag"]


    def init_from_msh(self, msh: Mesh, dtype: np.dtype = np.double, max_elements_to_process: int = 256):

        self.log.write("info", "Initializing the DirectSampler from a Mesh object")
        
        # Geometrical parameters for this mesh
        nelv = msh.nelv
        lz = msh.lz
        ly = msh.ly
        lx = msh.lx
        gdim = msh.gdim
        
        # Dictionary to store the settings as they are added
        self.settings = {}
        if dtype == np.float32:
            self.settings["dtype"] = "single"
        elif dtype == np.float64:
            self.settings["dtype"] = "double"
        self.settings["mesh_information"] = {"lx": lx, "ly": ly, "lz": lz, "nelv": nelv, "gdim": gdim}

        # Create a dictionary that will have the data that needs to be compressed later
        self.uncompressed_data = {}

        # Create a dictionary that will hold the data after compressed
        self.compressed_data = {}

        # Initialize the common parameters
        self.init_common(max_elements_to_process)

    def init_common(self, max_elements_to_process: int = 256):

        self.max_elements_to_process = max_elements_to_process

        # Mesh information
        self.lx = self.settings["mesh_information"]["lx"]
        self.ly = self.settings["mesh_information"]["ly"]
        self.lz = self.settings["mesh_information"]["lz"]
        self.gdim = self.settings["mesh_information"]["gdim"]
        self.nelv = self.settings["mesh_information"]["nelv"]

        # dtype
        if self.settings["dtype"] == "single":
            self.dtype = np.float32
        elif self.settings["dtype"] == "double":
            self.dtype = np.float64
        
        # Get transformation matrices for this mesh
        self.v, self.vinv, self.w3, self.x, self.w = get_transform_matrix(
            self.lx, self.gdim, apply_1d_operators=False, dtype=self.dtype
        )


    def clear(self):

        # Clear the data that has been sampled. This is necesary to avoid mixing things up when sampling new fields.
        self.settings = {}
        self.uncompressed_data = {}
        self.compressed_data = {}
    
    def sample_field(self, field: np.ndarray = None, field_name: str = "field", covariance_method: str = "average", covariance_elements_to_average: int = 1, covariance_keep_modes: int=1,
                    compression_method: str = "fixed_bitrate", bitrate: float = 1/2):
        
        self.log.write("info", "Sampling the field with options: covariance_method: {covariance_method}, compression_method: {compression_method}")

        # Copy the field into device if needed
        if self.bckend == "torch":
            field = torch.tensor(field, dtype=self.dtype_d, device = self.device, requires_grad=False)

        self.log.write("info", "Estimating the covariance matrix")
        self._estimate_field_covariance(field=field, field_name=field_name, method=covariance_method, elements_to_average=covariance_elements_to_average, keep_modes=covariance_keep_modes)

        if compression_method == "fixed_bitrate":
            self.settings["compression"] =  {"method": compression_method,
                                             "bitrate": bitrate,
                                             "n_samples" : int(self.lx*self.ly*self.lz * bitrate)}
            
            if self.bckend == "numpy":
                self.log.write("info", f"Sampling the field using the fixed bitrate method. using settings: {self.settings['compression']}")
                field_sampled = self._sample_fixed_bitrate(field, field_name, self.settings)
            elif self.bckend == "torch":
                self.log.write("info", f"Sampling the field using the fixed bitrate method. using settings: {self.settings['compression']}")
                self.log.write("info", f"Using backend: {self.bckend} on device: {self.device}")
                field_sampled = self._sample_fixed_bitrate_torch(field, field_name, self.settings)

            self.uncompressed_data[f"{field_name}"]["field"] = field_sampled
            self.log.write("info", f"Sampled_field saved in field uncompressed_data[\"{field_name}\"][\"field\"]")

        else:
            raise ValueError("Invalid method to sample the field")
        
    def compress_samples(self, lossless_compressor: str = "bzip2"):
        """
        """

        self.log.write("info", f"Compressing the data using the lossless compressor: {lossless_compressor}")
        self.log.write("info", "Compressing data in uncompressed_data")
        for field in self.uncompressed_data.keys():
            self.log.write("info", f"Compressing data for field [\"{field}\"]:")
            self.compressed_data[field] = {}
            for data in self.uncompressed_data[field].keys():
                self.log.write("info", f"Compressing [\"{data}\"] for field [\"{field}\"]")
                if self.bckend == "numpy":
                    self.compressed_data[field][data] = bz2.compress(self.uncompressed_data[field][data].tobytes())
                elif self.bckend == "torch":
                    self.compressed_data[field][data] = bz2.compress(self.uncompressed_data[field][data].cpu().numpy().tobytes())


    def write_compressed_samples(self, comm = None,  filename="compressed_samples.h5"):
        """
        Writes compressed data to an HDF5 file in a hierarchical format, with separate
        groups for each MPI rank. If parallel HDF5 is supported, all ranks write to a single file
        using the 'mpio' driver. Otherwise, a folder is created to hold separate files for each rank,
        and a log message is generated to indicate this behavior.
        
        Parameters:
            compressed_data (dict): A dictionary structured as { field: { data_key: compressed_bytes } }
            filename (str): Base filename for the HDF5 file.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        try:
            # Check if h5py was built with MPI support.
            if h5py.get_config().mpi:
                # Open a single file for parallel writing.
                f = h5py.File(filename, "w", driver="mpio", comm=comm)
            else:
                raise RuntimeError("Parallel HDF5 not supported in this h5py build.")
        except Exception:
            # Log that parallel HDF5 is not available and a folder will be created.
            self.log.write("info", "Parallel HDF5 not available; creating folder to store rank files.")
            base_name, _ = os.path.splitext(filename)
            folder_name = f"{base_name}_comp"
            if rank == 0:
                os.makedirs(folder_name, exist_ok=True)
            # Ensure all ranks wait until the folder has been created.
            comm.Barrier()
            file_path = os.path.join(folder_name, f"{base_name}_rank_{rank}.h5")
            f = h5py.File(file_path, "w")

        # Indicate that the data are bytes
        binary_dtype = h5py.vlen_dtype(np.uint8)

        with f:

            # If settings exist, add them as metadata in a top-level group.
            if hasattr(self, "settings") and self.settings is not None:
                # In parallel mode, have rank 0 create the settings group.
                if comm.Get_rank() == 0:
                    settings_group = f.create_group("settings")
                    settings_dict = {key: self.settings[key] for key in self.settings.keys() if key != "mesh_information"}
                    add_settings_to_hdf5(settings_group, settings_dict)
                                         

            # Ensure all ranks wait until settings are written.
            comm.Barrier()

            # Create a top-level group for this rank.
            rank_group = f.create_group(f"rank_{rank}")

            # Add the mesh information of the rank
            mesh_info_group = rank_group.create_group("mesh_information")
            add_settings_to_hdf5(mesh_info_group, self.settings["mesh_information"])

            for field, data_dict in self.compressed_data.items():
                # Create a subgroup for each field.
                field_group = rank_group.create_group(field)
                for data_key, compressed_bytes in data_dict.items():

                    # This step is necessary to convert the bytes to a numpy array. to store in HDF5 ...
                    # ... It produced problems until I did that.                    
                    data_array = np.frombuffer(compressed_bytes, dtype=np.uint8)
                    dset = field_group.create_dataset(data_key, (1,), dtype=binary_dtype)
                    dset[0] = data_array

    def read_compressed_samples(self, comm=None, filename="compressed_samples.h5"):
        """
        Reads an HDF5 file (or folder of files if non-parallel mode was used) created by write_compressed_samples.
        Assumes that the same number of ranks is used for reading as for writing and that each rank reads only its own data.
        
        Returns a tuple:
            (global_settings, local_data)
        where:
            - global_settings is a dictionary from the top-level "settings" group (e.g., with keys "covariance" and "compression")
            and augmented with the rank-specific "mesh_information".
            - local_data is a dictionary structured as { "compressed_data": { field: { data_key: compressed_bytes } } }
        """
        
        rank = comm.Get_rank()

        # Open the file in parallel mode if available; otherwise open the per-rank file.
        try:
            if h5py.get_config().mpi:
                f = h5py.File(filename, "r", driver="mpio", comm=comm)
                mode = "parallel"
            else:
                raise RuntimeError("Parallel HDF5 not supported")
        except Exception:
            base_name, _ = os.path.splitext(filename)
            folder_name = f"{base_name}_comp"
            file_path = os.path.join(folder_name, f"{base_name}_rank_{rank}.h5")
            f = h5py.File(file_path, "r")
            mode = "non_parallel"

        # Read global settings (from the top-level "settings" group, written by rank 0).
        global_settings = {}
        if rank == 0:
            global_settings = load_hdf5_settings(f["settings"])
        global_settings = comm.bcast(global_settings, root=0)

        # Read rank-specific data from the "rank_{rank}" group.
        rank_group = f[f"rank_{rank}"]

        # Read the rank-specific mesh information from the "mesh_information" subgroup.
        mesh_information = {}
        if "mesh_information" in rank_group:
            mesh_information = load_hdf5_settings(rank_group["mesh_information"])

        # Add mesh_information to global_settings.
        global_settings["mesh_information"] = mesh_information

        # Read compressed data from the remaining groups (fields).
        compressed_data = {}
        for field_key in rank_group:
            # Skip the mesh_information subgroup.
            if field_key == "mesh_information":
                continue
            field_group = rank_group[field_key]
            field_dict = {}
            for data_key in field_group:
                dset = field_group[data_key]
                # Each dataset is stored as an array of shape (1,) containing a variable-length uint8 array.
                field_dict[data_key] = dset[0].tobytes()
            compressed_data[field_key] = field_dict

        f.close()

        return global_settings, compressed_data
    
    def decompress_samples(self, settings, compressed_data=None):
        """
        Decompresses the compressed data in the compressed_data dictionary.
        """

        uncompressed_data = {}
        for field, data_dict in compressed_data.items():
            uncompressed_data[field] = {}
            for data_key, compressed_bytes in data_dict.items():

                dtype = settings["dtype"]

                # Select the shape based on the name of the data
                nelv = settings["mesh_information"]["nelv"]
                lz = settings["mesh_information"]["lz"]
                ly = settings["mesh_information"]["ly"]
                lx = settings["mesh_information"]["lx"]
                average = settings["covariance"]["averages"]
                elements_to_average = settings["covariance"]["elements_to_average"]
                keep_modes = settings["covariance"].get("keep_modes", 1)

                if data_key == "field":    
                    shape = (nelv, lz, ly, lx)
                elif data_key == "kw":
                    shape = (average, lx*ly*lz)
                elif data_key == "U":
                    shape = (average*elements_to_average, keep_modes)
                elif data_key == "s":
                    shape = (keep_modes)
                elif data_key == "Vt":
                    shape = (keep_modes, lx*ly*lz)
                else:
                    raise ValueError("Invalid data key")

                if dtype == "single":
                    temp = np.frombuffer(bz2.decompress(compressed_bytes), dtype=np.float32)
                elif dtype == "double":
                    temp = np.frombuffer(bz2.decompress(compressed_bytes), dtype=np.float64)

                uncompressed_data[field][data_key] = temp.reshape(shape)

        return uncompressed_data



    def _estimate_field_covariance(self, field: np.ndarray = None, field_name: str = "field", method="average", elements_to_average: int = 1, keep_modes: int = 1):
        """
        """

        # Create a dictionary to store the data that will be compressed
        self.uncompressed_data[f"{field_name}"] = {}

        self.log.write("info", "Transforming the field into to legendre space")
        field_hat = self.transform_field(field, to="legendre")
        # Temporary:
        self.field_hat = field_hat

        if method == "average":

            # In this case, the kw should be taken as already the diagonal form
            self.kw_diag = True

            self.settings["covariance"] = {"method": "average",
                                           "elements_to_average": elements_to_average,
                                           "averages": int(np.ceil(self.nelv/elements_to_average)),
                                           "kw_diag": self.kw_diag}

            self.log.write("info", f"Estimating the covariance matrix using the averaging method method. Averaging over {elements_to_average} elements at a time")
            kw = self._estimate_covariance_average(field_hat, self.settings["covariance"])

            # Store the covariances in the data to be compressed:
            self.uncompressed_data[f"{field_name}"]["kw"] = kw
            self.log.write("info", f"Covariance saved in field uncompressed_data[\"{field_name}\"][\"kw\"]")

        elif method == "svd":
            # In this case, the kw will not be only the diagonal of the stored data but an approximation of the actual covariance
            self.kw_diag = True
            
            self.settings["covariance"] = {"method": "svd",
                                           "averages" : self.nelv,
                                           "elements_to_average": int(1),
                                           "keep_modes": keep_modes,
                                           "kw_diag": self.kw_diag}
            
            self.log.write("info", f"Estimating the covariance matrix using the SVD method. Keeping {keep_modes} modes")
            U, s, Vt = self._estimate_covariance_svd(field_hat, self.settings["covariance"])

            # Store the covariances in the data to be compressed:
            self.uncompressed_data[f"{field_name}"]["U"] = U
            self.uncompressed_data[f"{field_name}"]["s"] = s
            self.uncompressed_data[f"{field_name}"]["Vt"] = Vt

            self.log.write("info", f"U saved in field uncompressed_data[\"{field_name}\"][\"U\"]")
            self.log.write("info", f"s saved in field uncompressed_data[\"{field_name}\"][\"s\"]")
            self.log.write("info", f"Vt saved in field uncompressed_data[\"{field_name}\"][\"Vt\"]")

        else:
            raise ValueError("Invalid method to estimate the covariance matrix")
    
    def _sample_fixed_bitrate(self, field: np.ndarray, field_name: str, settings: dict):
        """
        """
        
        # Set the reshaping parameters
        averages = settings["covariance"]["averages"]
        elements_to_average = settings["covariance"]["elements_to_average"]

        # Retrieve the number of samples
        n_samples = settings["compression"]["n_samples"]

        # Reshape the fields into the KW supported shapes
        y = field.reshape(averages, elements_to_average, field.shape[1], field.shape[2], field.shape[3])
        V = self.v
        numfreq = n_samples

        # Now reshape the x, y elements into column vectors
        y = field.reshape(averages, elements_to_average, field.shape[1] * field.shape[2] * field.shape[3], 1)

        #allocation the truncated field
        y_truncated = np.ones_like(y) * -50

        # Create an array that contains the indices of the elements that have been sampled
        # The first indext to store is always index 0
        ind_train = np.zeros((averages, elements_to_average, n_samples), dtype=int)

        # Set up some help for the selections
        avg_idx = np.arange(averages)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
        elem_idx = np.arange(elements_to_average)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

        chunk_size_e = self.max_elements_to_process
        n_chunks_e = int(np.ceil(elements_to_average / chunk_size_e))
        chunk_size_a = self.max_elements_to_process
        n_chunks_a = int(np.ceil(averages / chunk_size_a))

        for chunk_id_a in range(n_chunks_a):
            start_a = chunk_id_a * chunk_size_a
            end_a = (chunk_id_a + 1) * chunk_size_a
            if end_a > averages:
                end_a = averages

            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = (chunk_id_e + 1) * chunk_size_e
                if end_e > elements_to_average:
                    end_e = elements_to_average

                avg_idx = np.arange(start_a, end_a)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
                elem_idx = np.arange(start_e, end_e)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

                self.log.write("info",f"Proccesing up to {(avg_idx.flatten()[-1] + 1) * (elem_idx.flatten()[-1]+1)}/{self.nelv} element")

                avg_idx2 = avg_idx.reshape(avg_idx.shape[0], 1)
                elem_idx2 = elem_idx.reshape(1, elem_idx.shape[1])
                
                # Set the initial index to be added
                imax = np.zeros((avg_idx.shape[0], elem_idx.shape[1]), dtype=int)

                # Get covariance matrix
                kw = self._get_covariance_matrix(settings, field_name, avg_idx2, elem_idx2)

                for freq in range(0,numfreq):

                    self.log.write("debug", f"Obtaining sample {freq+1}/{numfreq}")

                    # Sort the indices for each average and element
                    ind_train[avg_idx2, elem_idx2, :freq+1] = np.sort(ind_train[avg_idx2, elem_idx2, :freq+1], axis=2)

                    # Get the prediction and the standard deviation
                    y_21, y_21_std = self.gaussian_process_regression(y, V, kw, ind_train, avg_idx, elem_idx, avg_idx2, elem_idx2, freq, predict_mean=False, predict_std=True) 

                    # Set the variance as zero for the samples that have already been selected
                    I, J, _ = np.ix_(np.arange(y_21_std.shape[0]), np.arange(y_21_std.shape[1]), np.arange(y_21_std.shape[2]))
                    I2, J2, K2 = np.ix_(avg_idx2.flatten(), elem_idx2.flatten(), np.arange(freq+1))
                    y_21_std[I, J, ind_train[I2,J2,K2]] = 0

                    # Get the index of the sample with the highest standardd deviation
                    imax = np.argmax(y_21_std, axis=2)
                    if np.any(imax == 0):
                        # Replace 0 values in imax with random integers not in ind_train
                        zero_indices = np.argwhere(imax == 0)
                        for i, j in zero_indices:
                            possible_indices = np.setdiff1d(np.arange(y.shape[2]), ind_train[avg_idx2[i, 0], elem_idx2[0, j], :freq+1])
                            if len(possible_indices) > 0:
                                imax[i, j] = np.random.choice(possible_indices)
 
                    # Assign the index to be added
                    if freq < numfreq-1:
                        ind_train[avg_idx2, elem_idx2,freq+1] = imax

                # Get the finally selected samples
                y_11 = y[avg_idx, elem_idx, ind_train[avg_idx2,elem_idx2,:],:]

                # This is still with column vectors at the end. We need to reshape it.
                y_truncated[avg_idx, elem_idx, ind_train[avg_idx2, elem_idx2, :],:] = y_11

                self.ind_train_sample = ind_train

        # Reshape the field back to its original shape
        return y_truncated.reshape(field.shape)

    def _sample_fixed_bitrate_torch(self, field: torch.Tensor, field_name: str, settings: dict):
        """
        """
        # Set the reshaping parameters
        averages = settings["covariance"]["averages"]
        elements_to_average = settings["covariance"]["elements_to_average"]
        n_samples = settings["compression"]["n_samples"]

        # For KW routines, we first reshape into a five-dimensional tensor.
        V = self.v_d
        numfreq = n_samples

        # Reshape the spatial dimensions into column vectors:
        # New shape: (averages, elements_to_average, spatial_elements, 1)
        y = field.reshape(averages, elements_to_average, field.shape[1] * field.shape[2] * field.shape[3], 1)

        # Allocate the truncated field with the same type and device as y, filled with -50.
        y_truncated = torch.ones_like(y) * -50

        # Create an array to hold indices of the sampled elements.
        ind_train = torch.zeros((averages, elements_to_average, n_samples), dtype=torch.int64, device=y.device)

        # Define index helpers (for the full range, though we will use slicing for subchunks)
        # (Not used in all places below since slicing simplifies contiguous chunks.)
        full_avg_idx = torch.arange(averages, device=y.device).view(averages, 1, 1)
        full_elem_idx = torch.arange(elements_to_average, device=y.device).view(1, elements_to_average, 1)

        # Set up chunking parameters to avoid processing too many elements at once.
        chunk_size_e = self.max_elements_to_process
        n_chunks_e = math.ceil(elements_to_average / chunk_size_e)
        chunk_size_a = self.max_elements_to_process
        n_chunks_a = math.ceil(averages / chunk_size_a)

        # Loop over chunks along the "averages" dimension.
        for chunk_id_a in range(n_chunks_a):
            start_a = chunk_id_a * chunk_size_a
            end_a = min((chunk_id_a + 1) * chunk_size_a, averages)

            # Loop over chunks along the "elements_to_average" dimension.
            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = min((chunk_id_e + 1) * chunk_size_e, elements_to_average)

                # Create chunk-specific index helpers.
                # avg_idx: shape (chunk_a, 1, 1); elem_idx: shape (1, chunk_e, 1)
                avg_idx = torch.arange(start_a, end_a, device=y.device).view(-1, 1, 1)
                elem_idx = torch.arange(start_e, end_e, device=y.device).view(1, -1, 1)
                # Also define flattened versions for later use.
                avg_idx2 = avg_idx.reshape(avg_idx.shape[0], 1)  # shape: (chunk_a, 1)
                elem_idx2 = elem_idx.reshape(1, elem_idx.shape[1])  # shape: (1, chunk_e)

                # Log processing info.
                last_avg = avg_idx.flatten()[-1].item() + 1
                last_elem = elem_idx.flatten()[-1].item() + 1
                self.log.write("info", f"Processing up to {last_avg * last_elem}/{self.nelv} element")

                # Initialize imax (for the indices with maximum standard deviation)
                imax = torch.zeros((avg_idx.shape[0], elem_idx.shape[1]), dtype=torch.int64, device=y.device)

                # Get the covariance matrix for the current chunk.
                print("Getting covariance matrix")
                kw = self._get_covariance_matrix(settings, field_name, avg_idx2, elem_idx2)

                # Loop over frequency/sample iterations.
                for freq in range(numfreq):

                    print('looping', freq, numfreq)
                    self.log.write("log", f"Obtaining sample {freq+1}/{numfreq}")

                    # For the current chunk, sort the sampled indices along the frequency axis.
                    # Using slicing over the contiguous block.
                    sub_ind_train = ind_train[start_a:end_a, start_e:end_e, :freq+1]
                    sorted_sub_ind, _ = torch.sort(sub_ind_train, dim=2)
                    ind_train[start_a:end_a, start_e:end_e, :freq+1] = sorted_sub_ind

                    # Get prediction and standard deviation from your Gaussian process regression.
                    # (Assumes that the called function accepts torch tensors and chunk indices.)
                    y_21, y_21_std = self.gaussian_process_regression_torch(
                        y, V, kw, ind_train,
                        avg_idx, elem_idx, avg_idx2, elem_idx2,
                        freq, predict_mean=False, predict_std=True
                    )

                    # Set the standard deviation to zero at indices that have already been sampled.
                    # For each (i, j) in the chunk and each k in 0..freq, set:
                    #    y_21_std[i, j, ind_train[i, j, k]] = 0
                    sub_y_std = y_21_std  # shape: (chunk_a, chunk_e, some_dim)
                    for k in range(freq + 1):
                        # Extract the sample indices for this frequency.
                        indices = ind_train[start_a:end_a, start_e:end_e, k]  # shape: (chunk_a, chunk_e)
                        # Create meshgrid for the first two dimensions.
                        a_idx, e_idx = torch.meshgrid(
                            torch.arange(end_a - start_a, device=y.device),
                            torch.arange(end_e - start_e, device=y.device),
                            indexing='ij'
                        )
                        sub_y_std[a_idx, e_idx, indices] = 0

                    # Get the index (along the third dimension) of the maximum standard deviation.
                    imax = torch.argmax(sub_y_std, dim=2)

                    # For any location where imax == 0, replace it with a random index not already sampled.
                    if (imax == 0).any():
                        zero_indices = (imax == 0).nonzero(as_tuple=False)
                        for idx in zero_indices:
                            i, j = idx[0].item(), idx[1].item()
                            # Get already selected indices for this (i,j) in the current chunk.
                            already_selected = ind_train[start_a:end_a, start_e:end_e, :freq+1][i, j].tolist()
                            all_indices = set(range(y.shape[2]))
                            possible_indices = list(all_indices - set(already_selected))
                            if possible_indices:
                                imax[i, j] = random.choice(possible_indices)

                    # Save the newly selected indices if more samples are needed.
                    if freq < numfreq - 1:
                        ind_train[start_a:end_a, start_e:end_e, freq+1] = imax

                # After finishing frequency iterations for this chunk, assign the final selected samples.
                # For each frequency k, we extract the corresponding sample from y and place it into y_truncated.
                # We work on the current chunk only.
                chunk_a = end_a - start_a
                chunk_e = end_e - start_e
                # Obtain sub-tensors for convenience.
                y_chunk = y[start_a:end_a, start_e:end_e, :, :]         # shape: (chunk_a, chunk_e, spatial_elements, 1)
                y_trunc_chunk = y_truncated[start_a:end_a, start_e:end_e, :, :]  # same shape
                # Create meshgrid for the two leading dimensions.
                a_idx, e_idx = torch.meshgrid(
                    torch.arange(chunk_a, device=y.device),
                    torch.arange(chunk_e, device=y.device),
                    indexing='ij'
                )
                for k in range(n_samples):
                    # For each (i,j) in the chunk, get the index from ind_train and assign that column.
                    indices = ind_train[start_a:end_a, start_e:end_e, k]  # shape: (chunk_a, chunk_e)
                    # Use advanced indexing to assign the corresponding values.
                    y_trunc_chunk[a_idx, e_idx, indices, :] = y_chunk[a_idx, e_idx, indices, :]
                # Place the updated chunk back into y_truncated.
                y_truncated[start_a:end_a, start_e:end_e, :, :] = y_trunc_chunk

                # Save the sampled indices in an attribute (if needed elsewhere).
                self.ind_train_sample = ind_train

        # Reshape the truncated field back to the original shape of "field"
        return y_truncated.reshape(field.shape)
 
    def reconstruct_field(self, field_name: str = None, get_mean: bool = True, get_std: bool = False):
        """
        """

        # Retrieve the sampled field
        sampled_field = self.uncompressed_data[field_name]["field"]
        settings = self.settings

        # Set the reshaping parameters
        averages = settings["covariance"]["averages"]
        elements_to_average = settings["covariance"]["elements_to_average"]

        # Retrieve the number of samples
        n_samples = settings["compression"]["n_samples"]

        # Reshape the fields into the KW supported shapes
        y = sampled_field.reshape(averages, elements_to_average, sampled_field.shape[1], sampled_field.shape[2], sampled_field.shape[3])
        V = self.v
        numfreq = n_samples

        # Now reshape the x, y elements into column vectors
        y = sampled_field.reshape(averages, elements_to_average, sampled_field.shape[1] * sampled_field.shape[2] * sampled_field.shape[3], 1)

        #allocation the truncated field
        y_reconstructed = None
        y_reconstructed_std = None
        if get_mean:
            y_reconstructed = np.ones_like(y) * -50
        if get_std:
            y_reconstructed_std = np.ones_like(y) * -50

        # Create an array that contains the indices of the elements that have been sampled
        # The first indext to store is always index 0
        ind_train = np.zeros((averages, elements_to_average, n_samples), dtype=int)
        ## Get the ind train from the sampled field
        for e in range(averages):
            for i in range(elements_to_average):
                temp = np.where(y[e,i] != -50)
                ind_train[e,i, :len(temp[0])] = temp[0]
        ind_train = np.sort(ind_train, axis=2)
        self.ind_train_rct = ind_train

        # Set up some help for the selections
        avg_idx = np.arange(averages)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
        elem_idx = np.arange(elements_to_average)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

        chunk_size_e = self.max_elements_to_process
        n_chunks_e = int(np.ceil(elements_to_average / chunk_size_e))
        chunk_size_a = self.max_elements_to_process
        n_chunks_a = int(np.ceil(averages / chunk_size_a))

        for chunk_id_a in range(n_chunks_a):
            start_a = chunk_id_a * chunk_size_a
            end_a = (chunk_id_a + 1) * chunk_size_a
            if end_a > averages:
                end_a = averages

            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = (chunk_id_e + 1) * chunk_size_e
                if end_e > elements_to_average:
                    end_e = elements_to_average

                avg_idx = np.arange(start_a, end_a)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
                elem_idx = np.arange(start_e, end_e)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

                self.log.write("info",f"Proccesing up to {(avg_idx.flatten()[-1] + 1) * (elem_idx.flatten()[-1]+1)}/{self.nelv} element")

                avg_idx2 = avg_idx.reshape(avg_idx.shape[0], 1)
                elem_idx2 = elem_idx.reshape(1, elem_idx.shape[1])
                
                # Get covariance matrix
                kw = self._get_covariance_matrix(settings, field_name, avg_idx2, elem_idx2)


                # Get the prediction and the standard deviation
                y_21, y_21_std = self.gaussian_process_regression(y, V, kw, ind_train, avg_idx, elem_idx, avg_idx2, elem_idx2, predict_mean=get_mean, predict_std=get_std) 

                # This is still with column vectors at the end. We need to reshape it.
                if get_mean:
                    y_reconstructed[avg_idx2, elem_idx2] = y_21
                if get_std:
                    y_reconstructed_std[avg_idx2, elem_idx2] = y_21_std.reshape(y_21_std.shape[0], y_21_std.shape[1], y_21_std.shape[2], 1)

        if get_mean:
            y_reconstructed = y_reconstructed.reshape(sampled_field.shape)
            y_reconstructed[sampled_field != -50] = sampled_field[sampled_field != -50]
        if get_std:
            y_reconstructed_std = y_reconstructed_std.reshape(sampled_field.shape)

        # Reshape the field back to its original shape
        return y_reconstructed, y_reconstructed_std
 
 
    def predict(self, field_sampled: np.ndarray = None):

        # Global allocation
        sampling_type = "max_ent"

        field_rct = np.zeros_like(self.field_hat, dtype=self.field_hat.dtype)

        for e in range(0,self.nelv):

            kw = self.kw[int(np.floor(e/self.elements_to_average))]
            x = field_sampled[e].reshape(-1,1)
            y = field_sampled[e].reshape(-1,1)

            y_lcl_rct,y_std_lcl_rct = lcl_predict(kw,self.v,self.n_samples,x,y,sampling_type, self.kw_diag)

            field_rct[e] = y_lcl_rct.reshape(field_rct[e].shape)
    
        return field_rct


    def lcl_sample(self, kw,V,numfreq,x,y,sampling_type):
        #local inputs
        V = V
        numfreq = numfreq
        
        x=x.reshape(-1,1)
        y=y.reshape(-1,1)

        #allocation
        y_lcl_trunc = np.ones(y.shape)*-50

        #Make Kw a matrix
        kw=np.diag(kw)

        # Some variables to loop over
        ind_train=[]
        imax=0

        for freq in range(0,numfreq):

            # Choose the sample index and sort previous ones
            ind_train.append(imax)
            ind_train.sort()

            # Find the indices of the entries that have not been sampled
            ind_notchosen=[]
            for i in range(0,y.shape[0]):
                if i not in ind_train:
                    ind_notchosen.append(i)

            x_11,y_11,k00,k11,k22,k02,k20,k12,k21 = get_samples_and_cov(x,y,V,kw,ind_train,ind_notchosen)

            y_21,y_21_std,imax = get_prediction_and_maxentropyindex(x_11,y_11,k00,k11,k22,k02,k20,k12,k21,ind_train)
                
        # This is a column vector 
        y_lcl_trunc[ind_train,0] = y_11

        return y_lcl_trunc

    def _estimate_covariance_average(self, field_hat : np.ndarray, settings: dict):

        if self.bckend == "numpy":
            # Retrieve the settings
            averages=settings["averages"]
            elements_to_average=settings["elements_to_average"]

            # Create an average of field_hat over the elements
            temp_field = field_hat.reshape(averages, elements_to_average, field_hat.shape[1], field_hat.shape[2], field_hat.shape[3])
            field_hat_mean = np.mean(temp_field, axis=1)        
            
            ### This block was to average with weights, but the coefficients do not really have that sort of mass matrix.
            ##temp_mass = self.B.reshape(averages, elements_to_average, self.B.shape[1], self.B.shape[2], self.B.shape[3])
            #
            ## Perform a weighted average with the mass matrix
            ##field_hat_mean = np.sum(temp_field * temp_mass, axis=1) / np.sum(temp_mass, axis=1)
            ###

            # This is the way in which I calculate the covariance here and then get the diagonals
            if self.kw_diag == True:
                # Get the covariances
                kw = np.einsum("eik,ekj->eij", field_hat_mean.reshape(averages,-1,1), field_hat_mean.reshape(averages,-1,1).transpose(0,2,1))

                # Extract only the diagonals
                kw = np.einsum("...ii->...i", kw)
            else:
                # But I can leave the calculation of the covariance itself for later and store here the average of field_hat
                kw = field_hat_mean.reshape(averages,-1,1)

        elif self.bckend == "torch":
            # Retrieve the settings
            averages=settings["averages"]
            elements_to_average=settings["elements_to_average"]

            # Create an average of field_hat over the elements
            temp_field = field_hat.reshape(averages, elements_to_average, field_hat.shape[1], field_hat.shape[2], field_hat.shape[3])
            field_hat_mean = torch.mean(temp_field, dim=1)        

            ### This block was to average with weights, but the coefficients do not really have that sort of mass matrix.
            ##temp_mass = self.B.reshape(averages, elements_to_average, self.B.shape[1], self.B.shape[2], self.B.shape[3])
            #
            ## Perform a weighted average with the mass matrix
            ##field_hat_mean = np.sum(temp_field * temp_mass, axis=1) / np.sum(temp_mass, axis=1)
            ###

            # This is the way in which I calculate the covariance here and then get the diagonals
            if self.kw_diag == True:
                # Get the covariances
                kw = torch.einsum("eik,ekj->eij", field_hat_mean.reshape(averages,-1,1), field_hat_mean.reshape(averages,-1,1).permute(0,2,1))

                # Extract only the diagonals
                kw = torch.einsum("...ii->...i", kw)
            else:
                # But I can leave the calculation of the covariance itself for later and store here the average of field_hat
                kw = field_hat_mean.reshape(averages,-1,1)
            
        return kw
    
    def _estimate_covariance_svd(self, field_hat : np.ndarray, settings: dict):

        if self.bckend == "numpy":
            # Retrieve the settings
            averages=settings["averages"] # In the case of SVD, this is 1
            elements_to_average=settings["elements_to_average"] # In the case of SVD, this is the number of elements in the rank.

            # Create a sort of snapshot matrix S = (Data in the element, element)
            S = np.copy(field_hat.reshape( averages * elements_to_average , field_hat.shape[1] * field_hat.shape[2] * field_hat.shape[3]))

            # Perform the SVD # It is likely that this should be done streaming / parallel
            U, s, Vt = np.linalg.svd(S, full_matrices=False)

            # Keep only the first keep_modes
            U = U[:, :settings["keep_modes"]]
            s = s[:settings["keep_modes"]]
            Vt = Vt[:settings["keep_modes"], :]

        elif self.bckend == "torch":

            # Retrieve the settings
            averages = settings["averages"]  # In the case of SVD, this is 1
            elements_to_average = settings["elements_to_average"]  # In the case of SVD, this is the number of elements in the rank.

            # Create a snapshot matrix S = (Data in the element, element)
            S = field_hat.reshape(averages * elements_to_average,
                                field_hat.shape[1] * field_hat.shape[2] * field_hat.shape[3]).clone()

            # Perform the SVD using torch.linalg.svd (set full_matrices=False to match NumPy's behavior)
            U, s, Vt = torch.linalg.svd(S, full_matrices=False)

            # Keep only the first keep_modes
            keep_modes = settings["keep_modes"]
            U = U[:, :keep_modes]
            s = s[:keep_modes]
            Vt = Vt[:keep_modes, :]            

        return U, s, Vt

 
    def transform_field(self, field: np.ndarray = None, to: str = "legendre") -> np.ndarray:
        """
        Transform the field to the desired space
        
        Args:
            field (np.ndarray): Field to be transformed
            to (str): Space to which the field will be transformed
        
        Returns:
            np.ndarray: Transformed field
        """

        if self.bckend == "numpy":
            if to == "legendre":
                return apply_operator(self.vinv, field)
            elif to == "physical":
                return apply_operator(self.v, field)
            else:
                raise ValueError("Invalid space to transform the field to")
        elif self.bckend == "torch":
            if to == "legendre":
                return torch_apply_operator(self.vinv_d, field)
            elif to == "physical":
                return torch_apply_operator(self.v_d, field)
            else:
                raise ValueError("Invalid space to transform the field to")


    def obs_sample_fixed_bitrate(self, n_samples: int):
        """
        """

        field_sampled = np.zeros_like(self.field_hat, dtype=self.field_hat.dtype)
        sampling_type = "max_ent"

        for e in range(0,self.nelv):

            kw = self.kw[int(np.floor(e/self.elements_to_average))]
            #x = self.transformed_index[e].reshape(-1,1)
            x = self.field[e].reshape(-1,1)
            y = self.field[e].reshape(-1,1)

            # The result is a column vector
            y_lcl_trunc = self.lcl_sample(kw,self.v,n_samples,x,y,sampling_type)

            field_sampled[e] = y_lcl_trunc.reshape(field_sampled[e].shape)

        self.field_sampled = field_sampled


    def _get_covariance_matrix(self, settings: dict, field_name: str, avg_idx2: np.ndarray, elem_idx2: np.ndarray):
        """
        """

        if self.bckend == "numpy":

            averages2 = avg_idx2.shape[0]
            elements_to_average2 = elem_idx2.shape[1]
            
            if settings["covariance"]["method"] == "average":
                if self.kw_diag == True:        
                    # Retrieve the diagonal of the covariance matrix
                    kw = self.uncompressed_data[f"{field_name}"]["kw"][avg_idx2[:,0]]
            
                    # Transform it into an actual matrix, not simply a vector
                    # Aditionally, add one axis to make it consistent with the rest of the arrays and enable broadcasting
                    kw_ = np.einsum('...i,ij->...ij', kw, np.eye(kw.shape[-1])).reshape(averages2, 1 ,  kw.shape[-1], kw.shape[-1])
                else:
                    # Retrieve the averaged hat fields
                    f_hat = self.uncompressed_data[f"{field_name}"]["kw"][avg_idx2[:,0]]
                    # Calculate the covariance matrix with f_hat@f_hat^T
                    kw = np.einsum("eik,ekj->eij", f_hat, f_hat.transpose(0,2,1))
                    # Add an axis to make it consistent with the rest of the arrays and enable broadcasting
                    kw_ = kw.reshape(kw.shape[0], 1, kw.shape[1], kw.shape[2])
                
                kw = kw_
            
            elif settings["covariance"]["method"] == "svd":

                self.log.write("debug", f"Obtaining the covariance matrix for the current chunk")

                # Retrieve the SVD components
                U = self.uncompressed_data[f"{field_name}"]["U"]
                s = self.uncompressed_data[f"{field_name}"]["s"]
                Vt = self.uncompressed_data[f"{field_name}"]["Vt"]

                # Select only the relevant entries of U
                ## Reshape to allow the indices to be broadcasted
                averages = settings["covariance"]["averages"]
                elements_to_average = settings["covariance"]["elements_to_average"]
                U = U.reshape(averages, elements_to_average, 1, -1) # Here I need to have ALL!
                ## Select the relevant entries
                U = U[avg_idx2, elem_idx2, :, :]
                #Reshape to original shape
                U = U.reshape(averages2*elements_to_average2, -1) # Here use the size of avrg_index and elem_index instead, since it is reduced.

                # Construct the f_hat
                f_hat = np.einsum("ik,k,kj->ij", U, s, Vt)

                # This is the way in which I calculate the covariance here and then get the diagonals
                if self.kw_diag == True:
                    # Get the covariances
                    kw_ = np.einsum("eik,ekj->eij", f_hat.reshape(averages2*elements_to_average2,-1,1), f_hat.reshape(averages2*elements_to_average2,-1,1).transpose(0,2,1))
                    
                    # Extract only the diagonals
                    kw_ = np.einsum("...ii->...i", kw_)
                    
                    # Transform it into an actual matrix, not simply a vector
                    # Aditionally, add one axis to make it consistent with the rest of the arrays and enable broadcasting
                    kw_ = np.copy(np.einsum('...i,ij->...ij', kw_, np.eye(kw_.shape[-1])))            

                    #Make the shape consistent
                    kw_ = kw_.reshape(averages2, elements_to_average2 ,  kw_.shape[-1], kw_.shape[-1])
                    
                else:
                    # Reshape
                    f_hat = f_hat.reshape(averages2*elements_to_average2,-1,1)
            
                    # Calculate the covariance matrix with f_hat@f_hat^T
                    kw_ = np.einsum("eik,ekj->eij", f_hat, f_hat.transpose(0,2,1))
            
                    # Add an axis to make it consistent with the rest of the arrays and enable broadcasting
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[1], kw_.shape[2])
                
                # Get the covariance matrix for the current chunk
                kw = np.copy(kw_)

        
        elif self.bckend == 'torch':

            averages2 = avg_idx2.shape[0]
            elements_to_average2 = elem_idx2.shape[1]

            if settings["covariance"]["method"] == "average":
                print('Getting covariance with average method')
                if self.kw_diag:
                    print('The kw is diagonal')
                    # Retrieve the diagonal of the covariance matrix
                    kw = self.uncompressed_data[f"{field_name}"]["kw"][avg_idx2[:, 0]]
                    # Transform it into an actual matrix (not just a vector) and add an extra axis
                    eye = torch.eye(kw.shape[-1], device=kw.device, dtype=kw.dtype)
                    kw_ = torch.einsum("...i,ij->...ij", kw, eye).reshape(averages2, 1, kw.shape[-1], kw.shape[-1])
                else:
                    # Retrieve the averaged hat fields
                    f_hat = self.uncompressed_data[f"{field_name}"]["kw"][avg_idx2[:, 0]]
                    # Calculate the covariance matrix as f_hat @ f_hat^T using einsum
                    kw = torch.einsum("eik,ekj->eij", f_hat, f_hat.permute(0, 2, 1))
                    # Add an axis to make it consistent for broadcasting
                    kw_ = kw.reshape(kw.shape[0], 1, kw.shape[1], kw.shape[2])
                
                kw = kw_

            elif settings["covariance"]["method"] == "svd":
                self.log.write("debug", f"Obtaining the covariance matrix for the current chunk")

                # Retrieve the SVD components
                U = self.uncompressed_data[f"{field_name}"]["U"]
                s = self.uncompressed_data[f"{field_name}"]["s"]
                Vt = self.uncompressed_data[f"{field_name}"]["Vt"]

                # Reshape U to allow broadcasting and select the relevant entries
                averages = settings["covariance"]["averages"]
                elements_to_average = settings["covariance"]["elements_to_average"]
                U = U.reshape(averages, elements_to_average, 1, -1)  # shape: (averages, elements_to_average, 1, -1)
                U = U[avg_idx2, elem_idx2, :, :]  # shape: (averages2, elements_to_average2, 1, -1)
                U = U.reshape(averages2 * elements_to_average2, -1)  # shape: (averages2*elements_to_average2, -1)

                # Construct f_hat using einsum. This performs the equivalent of U * diag(s) * Vt.
                f_hat = torch.einsum("ik,k,kj->ij", U, s, Vt)

                if self.kw_diag:
                    # Reshape f_hat so that each row becomes a matrix column vector
                    f_hat_reshaped = f_hat.reshape(averages2 * elements_to_average2, -1, 1)
                    # Compute the covariance matrices for each entry: f_hat @ f_hat^T
                    kw_ = torch.einsum("eik,ekj->eij", f_hat_reshaped, f_hat_reshaped.permute(0, 2, 1))
                    # Extract only the diagonals
                    kw_diag = torch.einsum("...ii->...i", kw_)
                    # Convert the diagonal vector into a full matrix by multiplying with an identity matrix
                    eye = torch.eye(kw_diag.shape[-1], device=kw_diag.device, dtype=kw_diag.dtype)
                    kw_ = torch.einsum("...i,ij->...ij", kw_diag, eye)
                    # Reshape so that the result has shape (averages2, elements_to_average2, n, n)
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[-2], kw_.shape[-1])
                else:
                    # Reshape f_hat and compute the covariance matrices
                    f_hat_reshaped = f_hat.reshape(averages2 * elements_to_average2, -1, 1)
                    kw_ = torch.einsum("eik,ekj->eij", f_hat_reshaped, f_hat_reshaped.permute(0, 2, 1))
                    # Add an axis for broadcasting and reshape accordingly
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[1], kw_.shape[2])
                
                # Ensure a copy is made if needed (torch.clone() is used in place of np.copy)
                kw = kw_.clone()
                
        print('Done with covariance!')

        return kw
    
    def gaussian_process_regression(self, y: np.ndarray, V: np.ndarray, kw: np.ndarray, 
                                    ind_train: np.ndarray, avg_idx: np.ndarray, elem_idx: np.ndarray,
                                    avg_idx2: np.ndarray, elem_idx2: np.ndarray, freq: int = None, predict_mean: bool = True, predict_std: bool = True):

        # select the correct freq index:
        if freq is None:
            freq_idex = slice(None)
        else:
            freq_idex = slice(freq+1)

        # Select the current samples            
        y_11 = y[avg_idx, elem_idx, ind_train[avg_idx2,elem_idx2,freq_idex],:]

        V_11 = V[ind_train[avg_idx2,elem_idx2,freq_idex], :]
        V_22 = V.reshape(1,1,V.shape[0],V.shape[1])

        ## Get covariance matrices
        ## This approach was faster than using einsum. Potentially due to the size of the matrices
        ### Covariance of the sampled entries
        temp = np.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
        k11 = np.matmul(temp, np.swapaxes(V_11, -1, -2))  # results in shape: (averages, elements_to_average, freq+1, freq+1)
        ### Covariance of the predictions
        temp = np.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
        k12 = np.matmul(temp, np.swapaxes(V_22, -1, -2))  # if V_22 is shaped appropriately
        k21 = k12.transpose(0, 1, 3, 2)

        temp = np.matmul(V_22, kw)  # shape: (averages, elements_to_average, n, n)
        k22 = np.matmul(temp, np.swapaxes(V_22, -1, -2))  # results in shape: (averages, elements_to_average, n, n)

        # Make predictions to sample
        ## Create some matrices to stabilize the inversions
        eps = 1e-10*np.eye(k11.shape[-1]).reshape(1,1,k11.shape[-1],k11.shape[-1])
        ## Predict the mean and covariance matrix of all entires (data set 2) given the known samples (data set 1)

        y_21 = None
        if predict_mean:
            y_21= k21@np.linalg.inv(k11+eps)@(y_11)

        y_21_std = None
        if predict_std:    
            sigma21 = k22 - (k21@np.linalg.inv(k11+eps)@k12)           
            ## Predict the standard deviation of all entires (data set 2) given the known samples (data set 1)
            y_21_std = np.sqrt(np.abs(np.einsum("...ii->...i", sigma21)))

        return y_21, y_21_std


    def gaussian_process_regression_torch(self, y: torch.Tensor, V: torch.Tensor, kw: torch.Tensor, 
                                    ind_train: torch.Tensor, avg_idx: torch.Tensor, elem_idx: torch.Tensor,
                                    avg_idx2: torch.Tensor, elem_idx2: torch.Tensor, 
                                    freq: int = None, predict_mean: bool = True, predict_std: bool = True):
        
        # Select the correct freq index:
        if freq is None:
            freq_idex = slice(None)
        else:
            freq_idex = slice(freq + 1)

        # Select the current samples (using advanced indexing)
        y_11 = y[avg_idx, elem_idx, ind_train[avg_idx2, elem_idx2, freq_idex], :]

        V_11 = V[ind_train[avg_idx2, elem_idx2, freq_idex], :]
        V_22 = V.reshape(1, 1, V.shape[0], V.shape[1])

        # Get covariance matrices
        # Covariance of the sampled entries
        temp = torch.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
        k11 = torch.matmul(temp, V_11.transpose(-1, -2))  # shape: (averages, elements_to_average, freq+1, freq+1)
        
        # Covariance for predictions
        temp = torch.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
        k12 = torch.matmul(temp, V_22.transpose(-1, -2))  # shape: (averages, elements_to_average, freq+1, n)
        k21 = k12.permute(0, 1, 3, 2)  # shape: (averages, elements_to_average, n, freq+1)

        temp = torch.matmul(V_22, kw)  # shape: (averages, elements_to_average, n, n)
        k22 = torch.matmul(temp, V_22.transpose(-1, -2))  # shape: (averages, elements_to_average, n, n)

        # Create a small epsilon for numerical stability in inversion.
        eps = 1e-10 * torch.eye(k11.shape[-1], device=k11.device, dtype=k11.dtype).reshape(1, 1, k11.shape[-1], k11.shape[-1])

        y_21 = None
        if predict_mean:
            # Predict the mean: y_21 = k21 * inv(k11 + eps) * y_11
            y_21 = torch.matmul(torch.matmul(k21, torch.linalg.inv(k11 + eps)), y_11)

        y_21_std = None
        if predict_std:
            # Predict the covariance of predictions
            sigma21 = k22 - torch.matmul(torch.matmul(k21, torch.linalg.inv(k11 + eps)), k12)
            # Extract the diagonal (variance) and compute the standard deviation
            y_21_std = torch.sqrt(torch.abs(torch.einsum("...ii->...i", sigma21)))

        return y_21, y_21_std

    def torch_gaussian_process_regression(self, y: np.ndarray, V: np.ndarray, kw: np.ndarray, 
                                    ind_train: np.ndarray, avg_idx: np.ndarray, elem_idx: np.ndarray,
                                    avg_idx2: np.ndarray, elem_idx2: np.ndarray, freq: int = None, predict_mean: bool = True, predict_std: bool = True,
                                    cholesky = True):

        # select the correct freq index:
        if freq is None:
            freq_idex = slice(None)
        else:
            freq_idex = slice(freq+1)

        # Select the current samples            
        y_11 = y[avg_idx, elem_idx, ind_train[avg_idx2,elem_idx2,freq_idex],:]

        V_11 = V[ind_train[avg_idx2,elem_idx2,freq_idex], :]
        V_22 = V.reshape(1,1,V.shape[0],V.shape[1])

        self.log.tic()
        if self.kw_diag == True:
        
            # Extract the diagonal entries from KW.
            # This results in an array of shape (averages, elements_to_average, n)
            kw_diag = np.diagonal(kw, axis1=-2, axis2=-1)

            # For V_11, which is assumed to have shape (averages, elements_to_average, freq+1, n),
            # multiplying by a diagonal matrix on the right is equivalent to element-wise scaling 
            # of its last axis. We add a new axis so that broadcasting works correctly.
            temp = V_11 * kw_diag[:, :, None, :]  
            # Compute k11: (averages, elements_to_average, freq+1, freq+1)
            k11 = np.matmul(temp, np.swapaxes(V_11, -1, -2))

            # Compute k12 similarly using V_22.
            k12 = np.matmul(temp, np.swapaxes(V_22, -1, -2))
            k21 = k12.transpose(0, 1, 3, 2)

            # For V_22, do the same diagonal multiplication.
            temp2 = V_22 * kw_diag[:, :, None, :]
            k22 = np.matmul(temp2, np.swapaxes(V_22, -1, -2))
        
        else:

            ## Get covariance matrices
            ## This approach was faster than using einsum. Potentially due to the size of the matrices
            ### Covariance of the sampled entries
            temp = np.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
            k11 = np.matmul(temp, np.swapaxes(V_11, -1, -2))  # results in shape: (averages, elements_to_average, freq+1, freq+1)
            ### Covariance of the predictions
            #temp = np.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
            k12 = np.matmul(temp, np.swapaxes(V_22, -1, -2))  # if V_22 is shaped appropriately
            k21 = k12.transpose(0, 1, 3, 2)

            temp = np.matmul(V_22, kw)  # shape: (averages, elements_to_average, n, n)
            k22 = np.matmul(temp, np.swapaxes(V_22, -1, -2))  # results in shape: (averages, elements_to_average, n, n)

        # Make predictions to sample
        ## Create some matrices to stabilize the inversions
        eps = 1e-10*np.eye(k11.shape[-1]).reshape(1,1,k11.shape[-1],k11.shape[-1])
        ## Predict the mean and covariance matrix of all entires (data set 2) given the known samples (data set 1)

        
        self.log.write("debug", "Calculated covariance")
        self.log.toc() 
        
        self.log.tic()
        if cholesky:

            #k11[np.where(k11 < 0)] = 0

            # Compute the Cholesky factor L such that (k11 + eps) = L L^T.
            L = np.linalg.cholesky(k11 + eps)  # L shape: (averages, elements_to_average, freq+1, freq+1)
            self.log.write("debug", "Cholesky factor computed")

            y_21 = None
            if predict_mean:
                # Solve (k11 + eps) * sol = y_11 using two triangular solves:
                # 1. Solve L * z = y_11:
                z = np.linalg.solve(L, y_11)
                # 2. Solve L^T * sol = z:
                sol = np.linalg.solve(np.swapaxes(L, -1, -2), z)
                # Predictive mean: y_21 = k21 @ sol
                y_21 = np.matmul(k21, sol)

            y_21_std = None
            if predict_std:
                # Instead of computing the full predictive covariance, we only need its diagonal.
                # Solve for v in (k11 + eps) * v = k12:
                self.log.write("info", f"solving L v = k12 with shape {L.shape} and {k12.shape}")
                #v = np.linalg.solve(L, k12)  # v shape: (averages, elements_to_average, freq+1, n)
                v = np.linalg.solve(L, k12)  # v shape: (a, b, m, n)
 
                self.log.write("debug", "v obtained")
                # The term k21*(k11+eps)^{-1}*k12 equals v^T v.
                # Its diagonal is obtained by summing the squares of v along the training dimension (axis 2).
                v_sq_sum = np.sum(v**2, axis=2)  # shape: (averages, elements_to_average, n)
                self.log.write("debug", "v^2 obtained")
                # Extract the diagonal of k22:
                batch0, batch1, n_test, _ = k22.shape
                diag_k22 = np.empty((batch0, batch1, n_test))
                for i in range(batch0):
                    for j in range(batch1):
                        diag_k22[i, j, :] = np.diagonal(k22[i, j, :, :])
                self.log.write("debug", "diag_k22 obtained")
                # The predictive variance (diagonal) is: diag(sigma21) = diag(k22) - v_sq_sum.
                sigma21_diag = diag_k22 - v_sq_sum
                self.log.write("debug", "sigma21_diag obtained")
                y_21_std = np.sqrt(np.abs(sigma21_diag))
                self.log.write("debug", "y_21_std obtained")
        
        else:

            y_21 = None
            if predict_mean:
                y_21= np.matmul(k21, np.linalg.solve(k11+eps, y_11))

            y_21_std = None
            if predict_std:    
                sigma21 = k22 - np.matmul(k21, np.linalg.solve(k11+eps ,k12))
                self.log.write("debug", "Predicted sigma21")           
                ## Predict the standard deviation of all entires (data set 2) given the known samples (data set 1)
                y_21_std = np.sqrt(np.abs(np.einsum("...ii->...i", sigma21)))
                self.log.write("debug", "Predicted y_21_std")

        self.log.write("debug", "Inverted")
        self.log.toc() 

        return y_21, y_21_std

def apply_operator(dr, field):
        """
        Apply a 2D/3D operator to a field
        """

        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        # ==================================================
        # Using loops
        # dudrst = np.zeros_like(field, dtype=field.dtype)
        # for e in range(0, nelv):
        #    tmp = field[e, :, :, :].reshape(-1, 1)
        #    dtmp = dr @ tmp
        #    dudrst[e, :, :, :] = dtmp.reshape((lz, ly, lx))
        # ==================================================

        # Using einsum
        field_shape = field.shape
        operator_shape = dr.shape
        field_shape_as_columns = (
            field_shape[0],
            field_shape[1] * field_shape[2] * field_shape[3],
            1,
        )

        # Reshape the field in palce
        field.shape = field_shape_as_columns

        # apply the 2D/3D operator broadcasting with einsum
        transformed_field = np.einsum(
            "ejk, ekm -> ejm",
            dr.reshape(1, operator_shape[0], operator_shape[1]),
            field,
        )

        # Reshape the field back to its original shape
        field.shape = field_shape
        transformed_field.shape = field_shape

        return transformed_field

def torch_apply_operator(dr, field):
    """
    Apply a 2D/3D operator to a field using PyTorch.
    
    Parameters:
      dr (torch.Tensor): The operator tensor with shape (N, N) where N = lz * ly * lx.
      field (torch.Tensor): The field tensor with shape (nelv, lz, ly, lx).
    
    Returns:
      torch.Tensor: The transformed field with the same shape as the input field.
    """
    # Save the original shape: (nelv, lz, ly, lx)
    original_shape = field.shape

    # Flatten the spatial dimensions: reshape to (nelv, lz*ly*lx, 1)
    field_flat = field.reshape(original_shape[0], -1, 1)

    # Prepare the operator for broadcasting by reshaping to (1, N, N)
    dr_reshaped = dr.reshape(1, dr.shape[0], dr.shape[1])

    # Apply the operator using einsum.
    # The einsum notation "ejk,ekm->ejm" indicates:
    # - 'e' indexes over the batch (nelv),
    # - 'j' indexes the output vector dimension,
    # - 'k' indexes the common dimension,
    # - 'm' is the singleton dimension.
    transformed_field = torch.einsum("ejk,ekm->ejm", dr_reshaped, field_flat)

    # Reshape the result back to the original field shape
    transformed_field = transformed_field.reshape(original_shape)
    
    return transformed_field

def get_samples_and_cov(x,y,V,kw,ind_train,ind_notchosen):

    # Sample the main signal
    x_11=x[ind_train,0]
    y_11=y[ind_train,0]

    # Sample the basis functions
    V_00 = V[ind_notchosen,:]
    V_11 = V[ind_train,:]
    V_22 = V

    # Get covariance matrices
    ## Covariance of not sampled entries
    k00 = V_00@kw@V_00.T
    k02 = V_00@kw@V_22.T
    k20=k02.T
    ## Covariance of the sampled entries
    k11 = V_11@kw@V_11.T
    k12 = V_11@kw@V_22.T
    k21=k12.T
    ## Covariance of the predictions
    k22 = V_22@kw@V_22.T

    return x_11,y_11,k00,k11,k22,k02,k20,k12,k21

def get_prediction_and_maxentropyindex(x_11,y_11,k00,k11,k22,k02,k20,k12,k21,ind_train):
    # Make predictions
    ## Create some matrices to stabilize the inversions
    eps = 1e-10*np.eye(len(ind_train))
    ## Predict the mean and covariance matrix of all entires (data set 2) given the known samples (data set 1)
    y_21= k21@np.linalg.inv(k11+eps)@(y_11)
    sigma21 = k22 - (k21@np.linalg.inv(k11+eps)@k12)
    ## Predict the standard deviation of all entires (data set 2) given the known samples (data set 1)
    y_21_std = np.sqrt(np.abs(np.diag(sigma21)))

    ## Calculate the mutual information
    ent=y_21_std

    ## The new sample is that one that has the higher mutual information
    for i in range(0,ent.shape[0]):
        if i in ind_train:
            ent[i]=0
    imax=np.argmax(ent)

    return y_21,y_21_std,imax

def lcl_predict(kw,V,numfreq,x,y,sampling_type, kw_diag=True):
##################### local subroutine ##################################
    #local inputs

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    #allocation
    y_lcl_rct = np.zeros(y.shape)

    if kw_diag == True:
        #Make Kw a matrix
        kw=np.diag(kw)
    else:
        kw = kw@kw.T
    #Make Kw a matrix
    #kw=np.diag(kw)

    ind_train=np.where(y!=-50)[0]

    # Find the indices of the entries that have not been sampled
    ind_notchosen=[]
    for i in range(0,y.shape[0]):
        if i not in ind_train:
            ind_notchosen.append(i)

    x_11,y_11,k00,k11,k22,k02,k20,k12,k21 = get_samples_and_cov(x,y,V,kw,ind_train,ind_notchosen)

    y_lcl_rct,y_std_lcl_rct,imax = get_prediction_and_maxentropyindex(x_11,y_11,k00,k11,k22,k02,k20,k12,k21,ind_train)

    y_lcl_rct[ind_train]=y_11

    return y_lcl_rct,y_std_lcl_rct

def add_settings_to_hdf5(h5group, settings_dict):
    """
    Recursively adds the key/value pairs from a settings dictionary to an HDF5 group.
    Dictionary values that are themselves dictionaries are added as subgroups;
    other values are stored as attributes.
    """
    for key, value in settings_dict.items():
        if isinstance(value, dict):
            subgroup = h5group.create_group(key)
            add_settings_to_hdf5(subgroup, value)
        else:
            h5group.attrs[key] = value

def load_hdf5_settings(group):
    """
    Recursively loads an HDF5 group into a dictionary.
    Attributes become key/value pairs and subgroups are loaded recursively.
    """
    settings = {}
    # Load attributes
    for key, value in group.attrs.items():
        settings[key] = value
    # Recursively load subgroups
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            settings[key] = load_hdf5_settings(item)
    return settings

def batched_triangular_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve L * X = B for X in a batched fashion.
    
    L: numpy array of shape (batch0, batch1, m, m)
    B: numpy array of shape (batch0, batch1, m, n_rhs)
    
    Returns:
      X: numpy array of shape (batch0, batch1, m, n_rhs)
    """
    # Convert to torch tensors (on CPU; you can also specify device='cuda' if available)
    L_t = torch.from_numpy(L).float()
    B_t = torch.from_numpy(B).float()
    # Use torch.linalg.solve which supports batched matrices.
    X_t = torch.linalg.solve(L_t, B_t)
    return X_t.cpu().numpy()