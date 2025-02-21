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

class DirectSampler:

    """ 
    Class to perform direct sampling on a field in the SEM format
    """

    def __init__(self, comm: MPI.Comm = None, dtype: np.dtype = np.double,  msh: Mesh = None, coef: Coef = None, max_elements_to_process: int = 256):
        
        self.log = Logger(comm=comm, module_name="DirectSampler")

        # Set a default parameter
        self.max_elements_to_process = max_elements_to_process

        # Geometrical parameters for this mesh
        self.nelv = msh.nelv
        self.lz = msh.lz
        self.ly = msh.ly
        self.lx = msh.lx

        # Save the mass matrix
        self.B = coef.B 

        # Some settings
        self.dtype = dtype

        # Get transformation matrices for this mesh
        self.v, self.vinv, self.w3, self.x, self.w = get_transform_matrix(
            msh.lx, msh.gdim, apply_1d_operators=False, dtype=dtype
        )

        # Dictionary to store the settings as they are added
        self.settings = {}
        self.settings["mesh_information"] = {"lx": self.lx, "ly": self.ly, "lz": self.lz, "nelv": self.nelv}

        # Create a dictionary that will have the data that needs to be compressed later
        self.data_to_compress = {}

        # Create a dictionary that will hold the data after compressed
        self.compressed_data = {}

    def clear(self):

        # Clear the data that has been sampled. This is necesary to avoid mixing things up when sampling new fields.
        self.settings = {}
        self.data_to_compress = {}
    
    def sample_field(self, field: np.ndarray = None, field_name: str = "field", covariance_method: str = "average", covariance_elements_to_average: int = 1, covariance_keep_modes: int=1,
                    compression_method: str = "fixed_bitrate", bitrate: float = 1/2):
        
        self.log.write("info", "Sampling the field with options: covariance_method: {covariance_method}, compression_method: {compression_method}")

        self.log.write("info", "Estimating the covariance matrix")
        self._estimate_field_covariance(field=field, field_name=field_name, method=covariance_method, elements_to_average=covariance_elements_to_average, keep_modes=covariance_keep_modes)

        if compression_method == "fixed_bitrate":
            self.settings["compression"] =  {"method": compression_method,
                                             "bitrate": bitrate,
                                             "n_samples" : int(self.lx*self.ly*self.lz * bitrate)}
            
            self.log.write("info", f"Sampling the field using the fixed bitrate method. using settings: {self.settings['compression']}")
            field_sampled = self._sample_fixed_bitrate(field, field_name, self.settings)

            self.data_to_compress[f"{field_name}"]["field"] = field_sampled
            self.log.write("info", f"Sampled_field saved in field data_to_compress[\"{field_name}\"][\"field\"]")

        else:
            raise ValueError("Invalid method to sample the field")
        
    def compress_samples(self, lossless_compressor: str = "bzip2"):
        """
        """

        self.log.write("info", f"Compressing the data using the lossless compressor: {lossless_compressor}")
        self.log.write("info", "Compressing data in data_to_compress")
        for field in self.data_to_compress.keys():
            self.log.write("info", f"Compressing data for field [\"{field}\"]:")
            self.compressed_data[field] = {}
            for data in self.data_to_compress[field].keys():
                self.log.write("info", f"Compressing [\"{data}\"] for field [\"{field}\"]")
                self.compressed_data[field][data] = bz2.compress(self.data_to_compress[field][data].tobytes())


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
                if comm.Get_rank == 0:
                    settings_group = f.create_group("settings")
                    add_settings_to_hdf5(settings_group, {"covariance": self.settings["covariance"],
                                                      "compression": self.settings["compression"]})

            # Ensure all ranks wait until settings are written.
            comm.Barrier()

            # Create a top-level group for this rank.
            rank_group = f.create_group(f"rank_{rank}")

            # Add the mesh information of the rank
            add_settings_to_hdf5(rank_group, self.settings["mesh_information"])

            for field, data_dict in self.compressed_data.items():
                # Create a subgroup for each field.
                field_group = rank_group.create_group(field)
                for data_key, compressed_bytes in data_dict.items():

                    # This step is necessary to convert the bytes to a numpy array. to store in HDF5 ...
                    # ... It produced problems until I did that.                    
                    data_array = np.frombuffer(compressed_bytes, dtype=np.uint8)
                    dset = field_group.create_dataset(data_key, (1,), dtype=binary_dtype)
                    dset[0] = data_array


    def _estimate_field_covariance(self, field: np.ndarray = None, field_name: str = "field", method="average", elements_to_average: int = 1, keep_modes: int = 1):
        """
        """

        # Create a dictionary to store the data that will be compressed
        self.data_to_compress[f"{field_name}"] = {}

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
            self.data_to_compress[f"{field_name}"]["kw"] = kw
            self.log.write("info", f"Covariance saved in field data_to_compress[\"{field_name}\"][\"kw\"]")
        
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
            self.data_to_compress[f"{field_name}"]["U"] = U
            self.data_to_compress[f"{field_name}"]["s"] = s
            self.data_to_compress[f"{field_name}"]["Vt"] = Vt

            self.log.write("info", f"U saved in field data_to_compress[\"{field_name}\"][\"U\"]")
            self.log.write("info", f"s saved in field data_to_compress[\"{field_name}\"][\"s\"]")
            self.log.write("info", f"Vt saved in field data_to_compress[\"{field_name}\"][\"Vt\"]")

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

        # Retrieve appropiate covariance outside the loop if needed:
        if settings["covariance"]["method"] == "average":
            if self.kw_diag == True:        
                # Retrieve the diagonal of the covariance matrix
                kw = self.data_to_compress[f"{field_name}"]["kw"]
        
                # Transform it into an actual matrix, not simply a vector
                # Aditionally, add one axis to make it consistent with the rest of the arrays and enable broadcasting
                kw_ = np.einsum('...i,ij->...ij', kw, np.eye(kw.shape[-1])).reshape(averages, 1 ,  kw.shape[-1], kw.shape[-1])
            else:
                # Retrieve the averaged hat fields
                f_hat = self.data_to_compress[f"{field_name}"]["kw"]
                # Calculate the covariance matrix with f_hat@f_hat^T
                kw = np.einsum("eik,ekj->eij", f_hat, f_hat.transpose(0,2,1))
                # Add an axis to make it consistent with the rest of the arrays and enable broadcasting
                kw_ = kw.reshape(kw.shape[0], 1, kw.shape[1], kw.shape[2])
           
        elif settings["covariance"]["method"] == "svd":
            pass

        else:
            raise ValueError("Invalid method to estimate the covariance matrix")

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
                averages2 = avg_idx2.shape[0]
                elements_to_average2 = elem_idx2.shape[1]

                # Set the initial index to be added
                imax = np.zeros((avg_idx.shape[0], elem_idx.shape[1]), dtype=int)

                # Retrieve also kw if the method requires it
                if settings["covariance"]["method"] == "average":
                    kw = kw_[avg_idx2[:,0]]

                elif settings["covariance"]["method"] == "svd":

                    self.log.write("info", f"Obtaining the covariance matrix for the current chunk")

                    # Retrieve the SVD components
                    U = self.data_to_compress[f"{field_name}"]["U"]
                    s = self.data_to_compress[f"{field_name}"]["s"]
                    Vt = self.data_to_compress[f"{field_name}"]["Vt"]

                    # Select only the relevant entries of U
                    ## Reshape to allow the indices to be broadcasted
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

                for freq in range(0,numfreq):

                    self.log.write("info", f"Obtaining sample {freq+1}/{numfreq}")

                    # Sort the indices for each average and element
                    ind_train[avg_idx2, elem_idx2, :freq+1] = np.sort(ind_train[avg_idx2, elem_idx2, :freq+1], axis=2)

                    # Select the current samples            
                    #x_11 = x[avg_idx, elem_idx, ind_train[:,:,:freq+1],:]
                    y_11 = y[avg_idx, elem_idx, ind_train[avg_idx2,elem_idx2,:freq+1],:]
        
                    V_11 = V[ind_train[avg_idx2,elem_idx2,:freq+1], :]
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
                    ## Actually do not predict the mean at this stage, only the covariance to save some time
                    #y_21= k21@np.linalg.inv(k11+eps)@(y_11)
                    sigma21 = k22 - (k21@np.linalg.inv(k11+eps)@k12)           
                    ## Predict the standard deviation of all entires (data set 2) given the known samples (data set 1)
                    y_21_std = np.sqrt(np.abs(np.einsum("...ii->...i", sigma21)))

                    # Set the variance as zero for the samples that have already been selected
                    #y_21_std[:, :, ind_train[avg_idx2,elem_idx2,:freq+1]] = 0

                    # Get the index of the sample with the highest standardd deviation
                    imax = np.argmax(y_21_std, axis=2)
                    
                    # Assign the index to be added
                    if freq < numfreq-1:
                        ind_train[avg_idx2, elem_idx2,freq+1] = imax

                # This is still with column vectors at the end. We need to reshape it.
                y_truncated[avg_idx, elem_idx, ind_train[avg_idx2, elem_idx2, :],:] = y_11

        # Reshape the field back to its original shape
        return y_truncated.reshape(field.shape)
 
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

        return kw
    
    def _estimate_covariance_svd(self, field_hat : np.ndarray, settings: dict):

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

        if to == "legendre":
            return apply_operator(self.vinv, field)
        elif to == "physical":
            return apply_operator(self.v, field)
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