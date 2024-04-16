import numpy as np

def generate_1d_arrays(bbox, n, mode="equal"):
    if mode=="equal":
        x_1d = np.linspace(bbox[0], bbox[1], n)
    elif mode=="cheb":
        x_1d = np.zeros(n)
        i = 0
        for k in range(1, n+1):
            # These are the chebyshev of the first kind (-1,1)
            #x_1d[i]=np.cos(((2*k-1)/(2*n))*np.pi)
            # These are the chebyshev of the second kind [-1,1]
            x_1d[i]= 1/2*(bbox[0]+bbox[1]) + 1/2*(bbox[1]-bbox[0]) * np.cos(((k-1)/(n-1))*np.pi)
            i = i+1
        x_1d = np.flip(x_1d)
    return x_1d  