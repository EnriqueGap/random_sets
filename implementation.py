import numpy as np
class RandomSets(object):
    
    def __init__(self,array):
        self.array=array
        self.bandwidth=1.
        self.bandwidth_matrix=[1,1]
    
    @property
    def bandwidth_matrix(self):
        return self._bandwidth_matrix
    
    @bandwidth_matrix.setter
    def bandwidth_matrix(self, value):
        if len(value)!=2:
            raise TypeError('Bandwidth matrix must contain two entries (h0, h1)')
        self._bandwidth_matrix = value
    
    def kernel_u(w):
        """
        univariate gaussian kernel
        
        Arguments:
        w :: float
        
        returns:
        float
        N(0,1)(w) mean zero unit variance normal distribution on w
        """
        return np.exp(-0.5*w**2)/np.sqrt(2*np.pi)
    
    def kernel_b(u,v):
        """
        bivariate gaussian kernel
        
        Arguments:
        u,v :: float
        
        returns:
        float
        N(0,0,1,1,0)(u,v) bivariate standard normal distribution on (u,v)
        """
        return np.exp(-0.5*(u**2 + v**2))/np.sqrt(2*np.pi)
    
    def d(x,y):
        """
        Metric function
        
        Arguments:
        x,y sets
        
        returns:
        float
        the distance between x and y elements of a metric space
        """
        return len(x.symmetric_difference(y))
    d_vec=np.vectorize(d)
    def marginal_est(self, X):
        """
        Marginal estimation
        
        Arguments:
        set
        X argument for estimation
        
        returns:
        float
        the marginal density estimation on X
        """
        arg=kernel_u(d_vec(X,self.array)/self.bandwidth)
        return np.sum(arg)/(len(self.array)*self.bandwidth)
    def density_est(self, X,Y):
        """
        density estimation
        
        Arguments:
        float
        X,Y vector for estimation
        
        returns:
        float
        the total density estimation on X,Y
        """
        array1=kernel_u(d_vec(X,self.array)/self.bandwidth_matrix[0])
        array2=kernel_u(d_vec(Y,self.array)/self.bandwidth_matrix[1])
        return np.dot(array1,array2)/(len(self.array)*self.bandwidth_matrix[0]*self.bandwidth_matrix[1])
