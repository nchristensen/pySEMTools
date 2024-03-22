import numpy as np
from scipy.spatial import KDTree

NoneType = type(None)

class msh_c():
    def __init__(self, comm, data = None, x = None, y = None, z = None):

        if type(data) != NoneType:
            self.x, self.y, self.z = get_coordinates_from_hexadata(data)
        else:
            self.x = x
            self.y = y
            self.z = z

        self.lx = self.x.shape[3] # This is not an error, the x data is on the last index
        self.ly = self.x.shape[2] # This is not an error, the x data is on the last index
        self.lz = self.x.shape[1] # This is not an error, the x data is on the last index
        self.lxyz = self.lx*self.ly*self.lz
        self.nelv = self.x.shape[0]
        
        #Find the element offset of each rank so you can store the global element number
        nelv = self.x.shape[0]
        sendbuf = np.ones((1), np.intc) * nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - nelv

        # Find the total number of elements
        sendbuf = np.ones((1), np.intc) * self.nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Allreduce(sendbuf, recvbuf)
        self.glb_nelv = recvbuf[0]

        if self.lz > 1:
            self.gdim = 3
        else:
            self.gdim = 2
 
        self.msh_list = np.zeros((self.nelv*self.lxyz, 3))
        self.linear_indices = []
        self.nonlinear_indices = []
        counter = 0
        for e in range(0, self.nelv):
            for k in range(0, self.lz):
                for j in range(0, self.ly):
                    for i in range(0, self.lx):
                        self.msh_list[linear_index(i,j,k,e, self.lx, self.ly, self.lz), 0] = self.x[e, k, j, i]
                        self.msh_list[linear_index(i,j,k,e, self.lx, self.ly, self.lz), 1] = self.y[e, k, j, i]
                        self.msh_list[linear_index(i,j,k,e, self.lx, self.ly, self.lz), 2] = self.z[e, k, j, i]
                        self.linear_indices.append([])
                        self.nonlinear_indices.append([])
                        
                        self.linear_indices[counter].append(linear_index(i,j,k,e, self.lx, self.ly, self.lz))
                        self.nonlinear_indices[counter] = nonlinear_index(self.linear_indices[counter], self.lx, self.ly, self.lz)
                        counter += 1


        self.my_tree = KDTree(self.msh_list)
        self.linear_shared_points = self.my_tree.query_ball_point(x=self.msh_list, r=1e-14, p=2.0, eps=0, workers=1, return_sorted=False, return_length=False)

        self.nonlinear_shared_points = []
        for i in range(0, len(self.linear_shared_points)):
            self.nonlinear_shared_points.append(nonlinear_index(self.linear_shared_points[i], self.lx, self.ly, self.lz))


        return 



def linear_index(i,j,k,l,lx,ly,lz):
    return (i + lx * ((j - 0) + ly * ((k - 0) + lz * ((l - 0)))))

def nonlinear_index(linear_index, lx,ly,lz):
    indices = []
    for list in linear_index:
        index = np.zeros(4,dtype=int)
        lin_idx = list
        index[3] = lin_idx/(lx*ly*lz)
        index[2] = (lin_idx-(lx*ly*lz)*index[3])/(lx*ly)
        index[1] = (lin_idx-(lx*ly*lz)*index[3]-(lx*ly)*index[2])/lx
        index[0] = (lin_idx-(lx*ly*lz)*index[3]-(lx*ly)*index[2]-lx*index[1])
        ind = (index[3], index[2], index[1], index[0])      
        indices.append(ind)
        
    return indices

def get_coordinates_from_hexadata(data):
    
    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    x = np.zeros((nelv,lz,ly,lx), dtype= np.double)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    for e in range(0,nelv):
        x[e,:,:,:] = data.elem[e].pos[0,:,:,:]
        y[e,:,:,:] = data.elem[e].pos[1,:,:,:]
        z[e,:,:,:] = data.elem[e].pos[2,:,:,:]

    return x, y, z
