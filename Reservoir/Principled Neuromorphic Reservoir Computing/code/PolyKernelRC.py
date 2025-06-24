# -*- coding: utf-8 -*-
"""

"""

import numpy as np

class PolyKernelRC():

    def __init__(
        self,
        system,
        k = 2,        
        taps = None,
        ridge_param_kern = 0,
        mask = [],
    ):
        super(PolyKernelRC, self).__init__()

        self.system = system
        # number of time delay taps
        self.k = k
        self.ridge_param_kern = ridge_param_kern
        self.mask = mask 

        # input dimension
        self.d=self.system.data.y.shape[0]        
        # size of linear part of feature vector
        self.dlin = self.k*self.d       
        if taps is None:
            self.taps = np.arange(0,k)
        else:
            self.taps = taps

        self.memoryBuffer()        
        self.data_kernel = np.transpose(self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1])
        self.target_kernel =  (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1])
        
        self.gram = self.kern_mat(self.data_kernel.T)

    # Compute NRMSE
    def nrmse(self, train = True):
        if train:
            nrmse_value = np.sqrt(np.mean((self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.prediction_train[:,:])**2)/self.system.total_var)    
        else:
            nrmse_value = np.sqrt(np.mean((self.buffer[0:self.d,self.system.warmtrain_pts-1:self.system.warmtrain_pts+self.system.lyaptime_pts-1]-self.prediction[0:self.d,0:self.system.lyaptime_pts])**2)/self.system.total_var)
        return nrmse_value 

    # Form the memory buffer
    def memoryBuffer(self):
        # create an array to hold the linear part of the feature vector
        self.buffer = np.zeros((self.dlin,self.system.maxtime_pts))
        
        # fill in the linear part of the feature vector for all times
        for delay in range(self.k):
            tap = self.taps[delay]
            for j in range(delay,self.system.maxtime_pts):
                self.buffer[self.d*delay:self.d*(delay+1),j]=self.system.data.y[:,j-tap]     

        
    # Specify encoding function for data samples
    def kern_mat(self, data):   
           
        gram_dp = self.data_kernel @ data
        kernels = np.zeros(gram_dp.shape)

        
        if self.mask[0] == 1:
            kernels += 1    
        
        if self.mask[1] == 1:
            kernels += gram_dp     
            
        for i in range(2,len(self.mask)):
            if self.mask[i] == 1:
                kernels += gram_dp**(i)
        
        return kernels

    # Train the classfier
    def fit(self):
        # Compute the classifier
        self.W_out_dual = self.target_kernel @ np.linalg.pinv(self.gram + self.ridge_param_kern*np.identity(np.size(self.gram,0)))
        
        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out_dual @ self.gram

        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()

    # Predict in the autoregressive mode
    def predict(self):
        # Create a place to store feature vectors for prediction
        self.prediction = np.zeros((self.dlin,self.system.testtime_pts)) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = self.buffer[:,self.system.warmtrain_pts-1]
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            # do a prediction
            k_kern = self.kern_mat(self.prediction[:,j])
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out_dual @ k_kern[:]

            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
                    

        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)        