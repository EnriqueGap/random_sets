from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
class RandomSets(object):
    def __init__(self,array, matrix=[1,1]):
        """
        Constructor method
        
        Args: array
        
        instance attributes:
        
        public array :: list
        public bandwidth_matrix :: np.array
        private vert and horz :: np.arrays reshaped
        private d_vec vectorized metric function
        """
        self.array=self._sort(array)
        self.bandwidth_matrix=matrix
        self.__vert=np.array(self.array).reshape(len(self.array),1)
        self.__horz=np.array(self.array).reshape(1,len(self.array))
        self._d_vec = np.vectorize(self._d)
        """
            vectorized metric function d(X,Y)

            Arguments:
            V,W sets or array of sets

            returns:
            array float
            the distance between Vith and Wjth entries for all i and j
        """
    """
    getter and setter for bandwidth_matrix
    We take care about the dimension and shape of bandwidth_matrix
    """
    @property
    def bandwidth_matrix(self):
        return self._bandwidth_matrix
    
    @bandwidth_matrix.setter
    def bandwidth_matrix(self, value):
        if len(value)!=2:
            raise TypeError('Bandwidth matrix must be a list with two float entries: [h0, h1]')
        self._bandwidth_matrix = value
    """
    Sorting sets and distributions
    """
    def _sort(self, array):
        for i in range(len(array)):
            idx = i
            for j in range(i+1, len(array)):
                if array[idx] > array[j]:
                    idx = j
            array[i], array[idx] = array[idx], array[i]
        return array
    """
    kernels and metric functions
    """
    def _kernel_u(self, x):
        """
        univariate gaussian kernel
    
        Arguments:
        x :: float
    
        returns:
        float
        N(0,1)(x) mean zero unit variance normal distribution on w
        """
        return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
    def _kernel_b(self, x,y):
        """
        bivariate gaussian kernel
    
        Arguments:
        x,y :: float
    
        returns:
        float
        N(0,0,1,1,0)(x,y) bivariate standard normal distribution on (u,v)
        """
        return np.exp(-0.5*(x**2 + y**2))/(2*np.pi)
    def _d(self, V,W):
        """
        Metric function
    
        Arguments:
        V,W sets
    
        returns:
        float
        the Fréchet-Nikodyn distance between V and W
        """
        return len(V.symmetric_difference(W)) 
    """
    distributions
    """   
    def density_est(self, V,W):
        """
        density function
        
        Arguments:
        V,W (array) sets
        
        Returns:
        density function of V,W
        (array) float    
        """
        array1=self._kernel_u(self._d_vec(V,self.__horz)/self.bandwidth_matrix[0])                       #pass the fixed V,W to the whole array of ordered sets
        array2=self._kernel_u(self._d_vec(self.__vert,W)/self.bandwidth_matrix[1])
        return np.dot(array1,array2)/(len(self.array)*self.bandwidth_matrix[0]*self.bandwidth_matrix[1]) # compute the sum using dot product
    def marginal_est(self, W):
        """
        marginal distribution
        
        Arguments:
        W set
        
        Return:
        marginal distribution of W
        """
        arr=self.density_est(self.__vert,W) #compute the array of the density estimations passing the whole array of sets to density_est
        return np.sum(arr)                  # sum entry by entry
    def conditional_prob(self, X,Y):
        """
        conditional probability
        
        Arguments:
        X,Y (array) sets
        
        Returns
        P(X|Y) the probability of X given Y
        (array) float
        """
        return self.density_est(X,Y)/self.marginal_est(Y)
    """
    plotting and export
    """
    def get_dist(self, W: set):
        """
        exports probability distribution
        Args: W set
        returns array with the conditional probability calculated over all sets
        """
        dist=self.conditional_prob(self.__vert,W).reshape(len(self.array))
        return dist
    def plot_dist(self, W: set):
        """
        conditional probability distribution

        Args: W set

        returns: plot of probability distribution
        """
        dist=self.conditional_prob(self.__vert,W)
        plt.plot(dist, label="P( |W)")
        plt.title("Conditional Probability Distribution")
        plt.xlabel("Distribution")
        plt.legend()
    def plot_density(self):
        """
        plot density distribution

        Args: none

        returns: plot of density distribution
        """
        plt.imshow(self.density_est(self.__vert,self.__horz), cmap='viridis', interpolation='nearest')
        plt.colorbar(shrink=0.8)
        plt.title("Density Distribution")

    def plot_distance(self, W: set):
        """
        plot the distance between W and the whole distribution of sets
        """
        distance=self._d_vec(self.__vert, W)
        plt.plot(distance)
        plt.title("Distance betweenn a sample set and each entry of the dataset")