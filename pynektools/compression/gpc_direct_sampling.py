""" Module that contains the class and methods to perform direct sampling on a field """

from mpi4py import MPI
from ..monitoring.logger import Logger
from ..datatypes.msh import Mesh
from ..datatypes.coef import Coef
from ..datatypes.coef import get_transform_matrix
import numpy as np
import sys

class DirectSampler:

    """ 
    Class to perform direct sampling on a field in the SEM format
    """

    def __init__(self, comm: MPI.Comm = None, msh: Mesh = None, coef: Coef = None, field: np.ndarray = None):
        
        self.log = Logger(comm=comm, module_name="DirectSampler")

        self.nelv = msh.nelv
        self.lz = msh.lz
        self.ly = msh.ly
        self.lx = msh.lx

        self.v, self.vinv, self.w3, self.x, self.w = get_transform_matrix(
            msh.lx, msh.gdim, apply_1d_operators=False, dtype=field.dtype
        )

        self.log.write("info", "Transforming the field into to legendre space")
        self.field_hat = self.transform_field(field, to="legendre")
        self.field = field

        self.log.write("info", "Filling array with linear coordinate per point in the elements")
        self.transformed_index  = np.zeros((field.shape[0], field.shape[1]*field.shape[2]*field.shape[3]), dtype=field.dtype)
        self.transformed_index[:] = np.linspace(1, field.shape[1]*field.shape[2]*field.shape[3], field.shape[1]*field.shape[2]*field.shape[3]) / (field.shape[1]*field.shape[2]*field.shape[3])
        self.transformed_index.shape = field.shape 

    def estimate_covariance(self, method="average", elements_to_average: int = 1, keep_modes: int = 1):
        """
        """

        if method == "average":
            self.covariance_method = "average"
            self.elements_to_average = elements_to_average
            self.averages=int(np.ceil(self.nelv/elements_to_average))

            self.log.write("info", f"Estimating the covariance matrix using the averaging method method. Averaging over {elements_to_average} elements at a time")
            self.kw = self._estimate_covariance_average(self.field_hat, elements_to_average)
        else:
            raise ValueError("Invalid method to estimate the covariance matrix")

    def sample(self , method: str = "fixed_bitrate", n_samples: int = 8): 
    
        if method == "fixed_bitrate":
            self.sampling_method = "fixed_bitrate"
            self.n_samples = n_samples
            self.log.write("info", f"Sampling the field using the fixed bitrate method. Sampling {n_samples} modes")
            self._sample_fixed_bitrate(n_samples)
        else:
            raise ValueError("Invalid method to sample the field")

#### ============================ START Rewrite from here

    def obs_sample_fixed_bitrate(self, n_samples: int):
        """
        """

        field_sampled = np.zeros_like(self.field_hat, dtype=self.field_hat.dtype)
        sampling_type = "max_ent"

        for e in range(0,self.nelv):

            kw = self.kw[int(np.floor(e/self.elements_to_average))]
            x = self.transformed_index[e].reshape(-1,1)
            y = self.field[e].reshape(-1,1)

            # The result is a column vector
            y_lcl_trunc = self.lcl_sample(kw,self.v,n_samples,x,y,sampling_type)

            field_sampled[e] = y_lcl_trunc.reshape(field_sampled[e].shape)

        self.field_sampled = field_sampled
    
    def _sample_fixed_bitrate(self, n_samples: int):
        """
        """

        # Reshape the fields into the KW supported shapes
        # Make kw a diagonal matrix, not only the arrays
        kw = np.einsum('...i,ij->...ij', self.kw, np.eye(self.kw.shape[-1])).reshape(self.averages, 1 ,  self.kw.shape[-1], self.kw.shape[-1])
        x = self.transformed_index.reshape(self.averages, self.elements_to_average, self.field.shape[1], self.field.shape[2], self.field.shape[3])
        y = self.field.reshape(self.averages, self.elements_to_average, self.field.shape[1], self.field.shape[2], self.field.shape[3])
        V = self.v
        numfreq = n_samples

        # Now reshape the x, y elements into column vectors
        x = self.transformed_index.reshape(self.averages, self.elements_to_average, self.field.shape[1] * self.field.shape[2] * self.field.shape[3], 1)
        y = self.field.reshape(self.averages, self.elements_to_average, self.field.shape[1] * self.field.shape[2] * self.field.shape[3], 1)

        #allocation the truncated field
        y_truncated = np.ones_like(y) * -50

        # Create an array that contains the indices of the elements that have been sampled
        ind_train = np.zeros((self.averages, self.elements_to_average, n_samples), dtype=int)
        imax = np.zeros((self.averages, self.elements_to_average), dtype=int)

        # Set up some help for the selections
        avg_idx = np.arange(self.averages)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
        elem_idx = np.arange(self.elements_to_average)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

        for freq in range(0,numfreq):

            # Assign the index to be added
            ind_train[:,:,freq] = imax

            # Sort the indices for each average and element
            ind_train[:, :, :freq+1] = np.sort(ind_train[:, :, :freq+1], axis=2)

            # Select the current samples            
            #x_11 = x[avg_idx, elem_idx, ind_train[:,:,:freq+1],:]
            y_11 = y[avg_idx, elem_idx, ind_train[:,:,:freq+1],:]
 
            V_11 = V[ind_train[:,:,:freq+1], :]
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

            # Get covariance matrices using einsum (Seems slower than what is above)
            ## Covariance of the sampled entries
            ##k11 = np.einsum('...ij,...jk,...kl->...il', V_11, kw, V_11.transpose(0,1,3,2), optimize=True)
            ##k12 = np.einsum('...ij,...jk,...kl->...il', V_11, kw, V_22.transpose(0,1,3,2), optimize=True)
            ##k21 = k12.transpose(0, 1, 3, 2)
            ## Covariance of the predictions
            ##k22 = np.einsum('...ij,...jk,...kl->...il', V_22, kw, V_22.transpose(0,1,3,2), optimize=True)

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
            y_21_std[avg_idx, elem_idx, ind_train[:,:,:freq+1]] = 0

            # Get the index of the sample with the highest standardd deviation
            imax = np.argmax(y_21_std, axis=2)

        # This is still with column vectors at the end. We need to reshape it.
        y_truncated[avg_idx, elem_idx, ind_train,:] = y_11

        # Reshape the field back to its original shape
        self.field_sampled = y_truncated.reshape(self.field.shape)
 
    def predict(self, field_sampled: np.ndarray = None):

        # Global allocation
        sampling_type = "max_ent"

        field_rct = np.zeros_like(self.field_hat, dtype=self.field_hat.dtype)

        for e in range(0,self.nelv):

            kw = self.kw[int(np.floor(e/self.elements_to_average))]
            x = self.transformed_index[e].reshape(-1,1)
            y = field_sampled[e].reshape(-1,1)

            y_lcl_rct,y_std_lcl_rct = lcl_predict(kw,self.v,self.n_samples,x,y,sampling_type)

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




#### ============================ STOP Rewrite from here


    def _estimate_covariance_average(self, field_hat, elements_to_average):

        # Determine the number of averages to be performed
        averages=self.averages

        # Create buffer to store the data
        kw_v=np.zeros((averages, self.field_hat.shape[-1]), dtype=self.field_hat.dtype)

        # Create an average of field_hat over the elements
        temp = field_hat.reshape(averages, elements_to_average, field_hat.shape[1], field_hat.shape[2], field_hat.shape[3])
        field_hat_mean = np.mean(temp, axis=1)        

        # Get the covariances
        kw = np.einsum("eik,ekj->eij", field_hat_mean.reshape(averages,-1,1), field_hat_mean.reshape(averages,-1,1).transpose(0,2,1))

        # Extract only the diagonals
        kw = np.einsum("...ii->...i", kw)

        return kw

    
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

def lcl_predict(kw,V,numfreq,x,y,sampling_type):
##################### local subroutine ##################################
    #local inputs

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    #allocation
    y_lcl_rct = np.zeros(y.shape)

    #Make Kw a matrix
    kw=np.diag(kw)

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