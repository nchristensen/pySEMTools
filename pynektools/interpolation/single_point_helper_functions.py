import numpy as np
from math import pi
from scipy.special import legendre

def apply_operators_3d(dr, ds, dt, x):

    'This function applies operators the same way as they are applied in NEK5000'
    'The only difference is that it is reversed, as this is python and we decided to leave that arrays as is'

    # Apply in r direction
    temp = x.reshape((int(x.size/dr.T.shape[0]), dr.T.shape[0])) @ dr.T

    # Apply in s direction
    temp = temp.reshape((ds.shape[1] , ds.shape[1], int(temp.size/(ds.shape[1]**2))))
    ### The nek5000 way uses a for loop
    ## temp2 = np.zeros((ds.shape[1], ds.shape[0], int(temp.size/(ds.shape[1]**2)))) # This is needed because dimensions could reduce
    ## for k in range(0, temp.shape[0]):
    ##     temp2[k,:,:] = ds@temp[k,:,:]
    ### We can do it optimized in numpy if we reshape the operator. This way it can broadcast
    temp = ds.reshape((1, ds.shape[0], ds.shape[1]))@temp
    
    # Apply in t direction
    temp = dt@temp.reshape(dt.shape[1], (int(temp.size/dt.shape[1])))
    
    return temp.reshape(-1,1)

def GLC_pwts(n):
    """ 
    Gauss-Lobatto-Chebyshev (GLC) points and weights over [-1,1]    
    Args: 
      `n`: int, number of nodes
    Returns 
       `x`: 1D numpy array of size `n`, nodes         
       `w`: 1D numpy array of size `n`, weights
    """
    def delt(i,n):
        del_=1.
        if i==0 or i==n-1:
           del_=0.5 
        return del_
    x=np.cos(np.arange(n)*pi/(n-1))
    w=np.zeros(n)    
    for i in range(n):
        tmp_=0.0
        for k in range(int((n-1)/2)):
            tmp_+=delt(2*k,n)/(1-4.*k**2)*np.cos(2*i*pi*k/(n-1))
        w[i]=tmp_*delt(i,n)*4/float(n-1)
    return x,w 

def GLL_pwts(n,eps=10**-8,maxIter=1000):
    """
    Generating `n `Gauss-Lobatto-Legendre (GLL) nodes and weights using the 
    Newton-Raphson iteration.
    Args:    
      `n`: int
         Number of GLL nodes
      `eps`: float (optional) 
         Min error to keep the iteration running
      `maxIter`: float (optional)
         Max number of iterations
    Outputs:
      `xi`: 1D numpy array of size `n`
         GLL nodes
      `w`: 1D numpy array of size `n`
         GLL weights
    Reference:
       Canuto C., Hussaini M. Y., Quarteroni A., Tang T. A., 
       "Spectral Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987.
       https://link.springer.com/book/10.1007/978-3-642-84108-8
    """
    V=np.zeros((n,n))  #Legendre Vandermonde Matrix
    #Initial guess for the nodes: GLC points
    xi,w_=GLC_pwts(n)
    iter_=0
    err=1000
    xi_old=xi
    while iter_<maxIter and err>eps:
        iter_+=1
        #Update the Legendre-Vandermonde matrix
        V[:,0]=1.
        V[:,1]=xi
        for j in range(2,n):
            V[:,j]=((2.*j-1)*xi*V[:,j-1] - (j-1)*V[:,j-2])/float(j)
        #Newton-Raphson iteration 
        xi=xi_old-(xi*V[:,n-1]-V[:,n-2])/(n*V[:,n-1])
        err=max(abs(xi-xi_old).flatten())
        xi_old=xi
    if (iter_>maxIter and err>eps):
       print('gllPts(): max iterations reached without convergence!')
    #Weights
    w=2./(n*(n-1)*V[:,n-1]**2.)
    return xi,w


def legendre_basis_at_xtest(n, xtest, use_scipy = False):

    m = len(xtest) # Number of points
    
    # Allocate space
    Leg=np.zeros((n,m))

    if not use_scipy:
        
        #First row is filled with 1 according to recursive formula
        Leg[0,:]=np.ones((1,m))
        #Second row is filled with x according to recursive formula
        Leg[1,:]=np.multiply(np.ones((1,m)),xtest)
    
        # Apply the recursive formula for all x_i
        #### THE ROWS HERE ARE THE ORDERS!
        # look for recursive formula here if you want to verify https://en.wikipedia.org/wiki/Legendre_polynomials
        for j in range(1,n-1):
            for k_ in range(0, m):
                Leg[j+1,k_]=((2*j+1)*xtest[k_]*Leg[j,k_]-j*Leg[j-1,k_])/(j+1)

    elif use_scipy:

        for j in range(0, n):
            Leg[j,:] = np.polyval(legendre(j), xtest)[:]
            #Leg[j,:] = np.polyval(self.legendre[j], xtest)[:]
    
    return Leg


def legendre_basis_derivative_at_xtest(legtest, xtest, use_scipy = False):

    ## Now find the derivative matrix D_N,ij=(dpi_j/dxi)_at_xi=xi_i
    ##https://en.wikipedia.org/wiki/Legendre_polynomials

    n = legtest.shape[0]
    m = legtest.shape[1]

    # Allocate space
    D_N=np.zeros((n,m))

    if not use_scipy:

        #First row is filled with 1 according to recursive formula
        D_N[0,:]=np.zeros((1,m))
        #Second row is filled with x according to recursive formula
        D_N[1,:]=np.ones((1,m))
    
        for j in range(1,n-1):
            for k_ in range(0, m):
                for p in range(j, 0 - 1, -2):
                    #if j==6 and k_==0: print(p)
                    D_N[j+1, k_] += 2*legtest[p, k_]/(np.sqrt(2/(2*p+1))**2)


    elif use_scipy:

        for j in range(0, n):
            D_N[j,:] = np.polyval(legendre(j).deriv(), xtest)[:]
            #D_N[j,:] = np.polyval(self.legendre_prm[j], xtest)[:]


    return D_N

def orthonormalize(Leg):

    n = Leg.shape[0]
    m = Leg.shape[1]
    Leg=Leg.T # nek and I transpose it for transform
    
    #Scaling factor as in books
    delta=np.ones(n)
    for i in range(0,n):
        #delta[i]=2/(2*i+1)       #it is the same both ways
        delta[i]=2/(2*(i+1)-1)
    delta[n-1]=2/(n-1)
    #print(delta)
    #Scaling factor to normalize
    for i in range(0,n):
        delta[i]=np.sqrt(1/delta[i])
        
    #apply the scaling factor
    for i in range(0,m):
        for j in range(0,n):
            Leg[i,j]=Leg[i,j]*delta[j]
    return Leg.T

def get_reference_element(self):

    n = self.n
    
    # Get the quadrature nodes
    x,w_=GLL_pwts(n) # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x_gll=np.flip(x)     # Quadrature
    w=np.flip(w_)    # Weights
 
    # Bounding boxes of the elements
    min_x = -1
    max_x = 1
    min_y = -1
    max_y = 1
    min_z = -1
    max_z = 1

    tmpx = np.zeros(len(x))
    tmpy = np.zeros(len(x))
    tmpz = np.zeros(len(x))
    for j in range(0,len(x)):
        tmpx[j]=((1-x_gll[j])/2*min_x+(1+x_gll[j])/2*max_x)
        tmpy[j]=((1-x_gll[j])/2*min_y+(1+x_gll[j])/2*max_y)
        tmpz[j]=((1-x_gll[j])/2*min_z+(1+x_gll[j])/2*max_z)

    x_3d=np.kron(np.ones((n)), np.kron(np.ones((n)), tmpx))
    y_3d=np.kron(np.ones((n)), np.kron(tmpy, np.ones((n))))
    z_3d=np.kron(tmpz, np.kron(np.ones((n)), np.ones((n))))        

    # Object atributes
    self.x_gll = x_gll
    self.w_gll = w
    self.x_e = x_3d.reshape(-1,1)
    self.y_e = y_3d.reshape(-1,1)
    self.z_e = z_3d.reshape(-1,1)
    
    return

## Create transformation matrices for the element (Only needs to be done once)
def get_legendre_transformation_matrices(self):
    
    ## Legendre basis at the element gll points
    leg_gll = legendre_basis_at_xtest(self.n, self.x_gll)
    leg_prm_gll = legendre_basis_derivative_at_xtest(leg_gll, self.x_gll)
    
    ### Transformation matrices for the element (1D)
    v_1d = leg_gll.T
    v_1d_inv = np.linalg.inv(v_1d)
    d_1d = leg_prm_gll.T
    ### Transformation matrices in 2d
    v_2d = np.kron(v_1d,v_1d)
    v_2d_inv = np.kron(v_1d_inv,v_1d_inv)
    ### Transformation matrices in 3d
    v_3d = np.kron(v_1d,v_2d)
    v_3d_inv = np.kron(v_1d_inv,v_2d_inv)

    self.v1d = v_1d
    self.v1d_inv = v_1d_inv

    # Assign attributes
    self.v = v_3d
    self.v_inv = v_3d_inv

    return 

#Standard Lagrange interpolation
def lagInterp_matrix_at_xtest(x,xTest):
    """
    Lagrange interpolation in 1D space
    """
    n=len(x)
    m=len(xTest)
    k=np.arange(n)
    Lk=np.zeros((n,m))
    for k_ in k:
        prod_=1.0
        for j in range(n):
            if j!=k_:
                prod_*=(xTest-x[j])/(x[k_]-x[j])
        Lk[k_,:]=prod_  
        
    return Lk

#Standard Lagrange interpolation
def lagInterp_derivative_matrix_at_xtest(x, Lk, xTest):
    """
    Lagrange interpolation in 1D space
    """
    n=len(x) 
    k=np.arange(n) # k and n are the same range
    m=len(xTest) 

    Lk_sum=np.zeros((n,m))
    for k_ in k:
        suma = 0.0
        for j in range(n):
            if j!=k_:
                suma += 1/(xTest-x[j])
        Lk_sum[k_,:]=suma
 
    Lk_prm = np.multiply(Lk_sum,Lk)

    return Lk_prm