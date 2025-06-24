# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from scipy import sparse
import torch

# Initial version of this function comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code

class ProductRepresentationRC():

    def __init__(
        self,
        system,
        k = 2,
        taps = None,
        ridge_param = 0,
        bias = True,
    ):
        super(ProductRepresentationRC, self).__init__()
        
        self.system = system
        # number of time delay taps
        self.k = k
        self.ridge_param = ridge_param
        self.bias = bias
        # input dimension
        self.d=self.system.data.y.shape[0]        
        # size of linear part of feature vector
        self.dlin = self.k*self.d       
        if taps is None:
            self.taps = np.arange(0,k)
        else:
            self.taps = taps
            
        self.memoryBuffer()

    # Form the memory buffer
    def memoryBuffer(self):
        # create an array to hold the linear part of the feature vector
        self.buffer = np.zeros((self.dlin,self.system.maxtime_pts))
        
        # fill in the linear part of the feature vector for all times
        for delay in range(self.k):
            tap = self.taps[delay]
            for j in range(delay,self.system.maxtime_pts):
                self.buffer[self.d*delay:self.d*(delay+1),j]=self.system.data.y[:,j-tap]        

    # Compute NRMSE
    def nrmse(self, train = True):
        if train:
            nrmse_value = np.sqrt(np.mean((self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.prediction_train[:,:])**2)/self.system.total_var)    
        else:
            nrmse_value = np.sqrt(np.mean((self.buffer[0:self.d,self.system.warmtrain_pts-1:self.system.warmtrain_pts+self.system.lyaptime_pts-1]-self.prediction[0:self.d,0:self.system.lyaptime_pts])**2)/self.system.total_var)
        return nrmse_value 

    # Train the classfier for 1- & 2- order features
    def fit12(self):        
        # Compute the features
        # size of nonlinear part of feature vector
        self.dnonlin = int(self.dlin*(self.dlin+1)/2)
        # total size of feature vector: bias [possbily] + linear + nonlinear
        self.dtot = int(self.bias) + self.dlin + self.dnonlin      
        
        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        self.out_train = np.ones((self.dtot,self.system.traintime_pts))
        
        # copy over the linear part (shift over by one to account for constant)
        self.out_train[int(self.bias):self.dlin+int(self.bias),:]=self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                
        # fill in the non-linear part
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                self.out_train[self.dlin+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                cnt += 1
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]) @ self.out_train[:,:].T @ np.linalg.pinv(self.out_train[:,:] @ self.out_train[:,:].T +self.ridge_param*np.identity(self.dtot))

        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.out_train[:,0:self.system.traintime_pts]

        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()

    # Train the classfier for 1- & 3- order features
    def fit13(self):        
        # Compute the features
        # size of nonlinear part of feature vector
        self.dnonlin = int(self.dlin*(self.dlin+1)*(self.dlin+2)/6)
        # total size of feature vector: bias [possbily] + linear + nonlinear
        self.dtot = int(self.bias) + self.dlin + self.dnonlin      
        
        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        self.out_train = np.ones((self.dtot,self.system.traintime_pts))
        
        # copy over the linear part (shift over by one to account for constant)
        self.out_train[int(self.bias):self.dlin+int(self.bias),:]=self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                
        # fill in the non-linear part
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                for span in range(column,self.dlin):
                    self.out_train[self.dlin+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[span,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                    cnt += 1
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]) @ self.out_train[:,:].T @ np.linalg.pinv(self.out_train[:,:] @ self.out_train[:,:].T +self.ridge_param*np.identity(self.dtot))

        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.out_train[:,0:self.system.traintime_pts]

        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()   
                        
    # Train the classfier for 1-, 2- & 3- order features
    def fit123(self):        
        # Compute the features
        # size of nonlinear part of feature vector        
        self.dnonlin2 = int(self.dlin*(self.dlin+1)/2) 
        self.dnonlin3 = int(self.dlin*(self.dlin+1)*(self.dlin+2)/6)
        # total size of feature vector: bias [possbily] + linear + nonlinear
        self.dtot = int(self.bias) + self.dlin + self.dnonlin2 +  self.dnonlin3     
        
        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        self.out_train = np.ones((self.dtot,self.system.traintime_pts))
        
        # copy over the linear part (shift over by one to account for constant)
        self.out_train[int(self.bias):self.dlin+int(self.bias),:]=self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1]

        # fill in the non-linear part - 2nd order
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                self.out_train[self.dlin+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                cnt += 1
                
        # fill in the non-linear part - 3rd order
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                for span in range(column,self.dlin):
                    self.out_train[self.dlin+self.dnonlin2+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[span,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                    cnt += 1
       
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]) @ self.out_train[:,:].T @ np.linalg.pinv(self.out_train[:,:] @ self.out_train[:,:].T +self.ridge_param*np.identity(self.dtot))

        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.out_train[:,0:self.system.traintime_pts]

        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()      
          
    # Predict in the autoregressive mode
    def predict12(self):
        # Create a place to store feature vectors for prediction
        self.out_test = np.ones(self.dtot)              # full feature vector
        self.prediction = np.zeros((self.dlin,self.system.testtime_pts)) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = self.buffer[:,self.system.warmtrain_pts-1]
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            # copy linear part into whole feature vector
            self.out_test[int(self.bias):self.dlin+int(self.bias)]=self.prediction[:,j] 
            # fill in the non-linear part
            cnt=0
            for row in range(self.dlin):
                for column in range(row,self.dlin):
                    self.out_test[self.dlin+cnt+int(self.bias)]=self.prediction[row,j]*self.prediction[column,j]
                    cnt += 1

            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ self.out_test[:]

            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
                    
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      
        
    def predict13(self):
        # Create a place to store feature vectors for prediction
        self.out_test = np.ones(self.dtot)              # full feature vector
        self.prediction = np.zeros((self.dlin,self.system.testtime_pts)) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = self.buffer[:,self.system.warmtrain_pts-1]
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            # copy linear part into whole feature vector
            self.out_test[int(self.bias):self.dlin+int(self.bias)]=self.prediction[:,j] 
            # fill in the non-linear part
            cnt=0
            for row in range(self.dlin):
                for column in range(row,self.dlin):
                    for span in range(column,self.dlin):
                        self.out_test[self.dlin+cnt+int(self.bias)]=self.prediction[row,j]*self.prediction[column,j]*self.prediction[span,j]
                        cnt += 1
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ self.out_test[:]

            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      
        
    def predict123(self):
        # Create a place to store feature vectors for prediction
        self.out_test = np.ones(self.dtot)              # full feature vector
        self.prediction = np.zeros((self.dlin,self.system.testtime_pts)) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = self.buffer[:,self.system.warmtrain_pts-1]
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            # copy linear part into whole feature vector
            self.out_test[int(self.bias):self.dlin+int(self.bias)]=self.prediction[:,j] 
            # fill in the non-linear part  - 2nd order
            cnt=0
            for row in range(self.dlin):
                for column in range(row,self.dlin):
                    self.out_test[self.dlin+cnt+int(self.bias)]=self.prediction[row,j]*self.prediction[column,j]
                    cnt += 1
            # fill in the non-linear part  - 3rd order
            cnt=0
            for row in range(self.dlin):
                for column in range(row,self.dlin):
                    for span in range(column,self.dlin):
                        self.out_test[self.dlin+self.dnonlin2+cnt+int(self.bias)]=self.prediction[row,j]*self.prediction[column,j]*self.prediction[span,j]
                        cnt += 1
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ self.out_test[:]

            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      



class ProductRepresentationRCTorch():
    def __init__(
        self,
        system,
        device,
        k = 2,
        taps = None,
        ridge_param = 0,
        bias = True,
    ):
        super(ProductRepresentationRCTorch, self).__init__()
        
        self.system = system
        self.device = device
        # number of time delay taps
        self.k = k
        self.ridge_param = ridge_param
        self.bias = bias
        # input dimension
        self.d=self.system.data.y.shape[0]        
        # size of linear part of feature vector
        self.dlin = self.k*self.d       
        if taps is None:
            self.taps = torch.arange(0,k)
        else:
            self.taps = taps      
        self.memoryBuffer()

        self.indexing_2nd=torch.zeros((self.dlin,self.dlin), dtype=torch.bool, device=self.device)
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                self.indexing_2nd[row,column]=True

        self.indexing_3rd=torch.zeros((self.dlin,self.dlin,self.dlin), dtype=torch.bool, device=self.device)
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                for span in range(column,self.dlin):
                    self.indexing_3rd[row,column,span]=True

        self.indexing_4th=torch.zeros((self.dlin,self.dlin,self.dlin,self.dlin), dtype=torch.bool, device=self.device)
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                for span in range(column,self.dlin):
                    for span4 in range(span,self.dlin):
                        self.indexing_4th[row,column,span,span4]=True
    
    # Form the memory buffer
    def memoryBuffer(self):
        # create an array to hold the linear part of the feature vector
        self.buffer = torch.zeros((self.dlin,self.system.maxtime_pts), dtype=torch.float64, device=self.device)
        
        # fill in the linear part of the feature vector for all times
        for delay in range(self.k):
            tap = self.taps[delay]
            for j in range(delay,self.system.maxtime_pts):
                self.buffer[self.d*delay:self.d*(delay+1),j]=torch.clone(self.system.data.y[:,j-tap])        

    # Compute NRMSE
    def nrmse(self, train = True):
        if train:
            nrmse_value = torch.sqrt(torch.mean((self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.prediction_train[:,:])**2)/self.system.total_var)    
        else:
            nrmse_value = torch.sqrt(torch.mean((self.buffer[0:self.d,self.system.warmtrain_pts-1:self.system.warmtrain_pts+self.system.lyaptime_pts-1]-self.prediction[0:self.d,0:self.system.lyaptime_pts])**2)/self.system.total_var)
        return nrmse_value 

    # Compute readout
    def readout(self):        
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = torch.linalg.lstsq(self.out_train[:,:] @ self.out_train[:,:].T +self.ridge_param*torch.eye(self.dtot, dtype=torch.float64, device=self.device), self.out_train[:,:] @  (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]).T, driver='gels').solution.T

        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.out_train[:,0:self.system.traintime_pts]
    # Compute reservoir with 1st order features
    def reservoir1(self):
        # Compute the features
        # total size of feature vector: bias [possbily] + linear + nonlinear
        self.dtot = int(self.bias) + self.dlin 
        
        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        self.out_train = torch.ones((self.dtot,self.system.traintime_pts), dtype=torch.float64, device=self.device)
                
        # copy over the linear part (shift over by one to account for constant)
        self.out_train[int(self.bias):self.dlin+int(self.bias),:]=torch.clone(self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1])
                
    # Compute reservoir with 1st & 2nd order features
    def reservoir12(self):
        # Compute the features
        # size of nonlinear part of feature vector
        self.dnonlin = int(self.dlin*(self.dlin+1)/2)
        # total size of feature vector: bias [possbily] + linear + nonlinear
        self.dtot = int(self.bias) + self.dlin + self.dnonlin      
        
        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        self.out_train = torch.ones((self.dtot,self.system.traintime_pts), dtype=torch.float64, device=self.device)
                
        # copy over the linear part (shift over by one to account for constant)
        self.out_train[int(self.bias):self.dlin+int(self.bias),:]=torch.clone(self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1])
                
        # fill in the non-linear part
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                self.out_train[self.dlin+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                cnt += 1

    # Compute reservoir with 1st & 3rd order features
    def reservoir13(self):
        # Compute the features
        # size of nonlinear part of feature vector
        self.dnonlin = int(self.dlin*(self.dlin+1)*(self.dlin+2)/6)
        # total size of feature vector: bias [possbily] + linear + nonlinear
        self.dtot = int(self.bias) + self.dlin + self.dnonlin      
        
        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        self.out_train = torch.ones((self.dtot,self.system.traintime_pts), dtype=torch.float64, device=self.device)
                
        # copy over the linear part (shift over by one to account for constant)
        self.out_train[int(self.bias):self.dlin+int(self.bias),:]=torch.clone(self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1])
                
        # fill in the non-linear part
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                for span in range(column,self.dlin):
                    self.out_train[self.dlin+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[span,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                    cnt += 1

    # Compute reservoir with 1st, 2nd & 3rd order features
    def reservoir123(self):
        # Compute the features
        # size of nonlinear part of feature vector
        self.dnonlin2 = int(self.dlin*(self.dlin+1)/2)
        self.dnonlin3 = int(self.dlin*(self.dlin+1)*(self.dlin+2)/6)
        # total size of feature vector: bias [possbily] + linear + nonlinear
        self.dtot = int(self.bias) + self.dlin + self.dnonlin2 +  self.dnonlin3  
        
        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        self.out_train = torch.ones((self.dtot,self.system.traintime_pts), dtype=torch.float64, device=self.device)
                
        # copy over the linear part (shift over by one to account for constant)
        self.out_train[int(self.bias):self.dlin+int(self.bias),:]=torch.clone(self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1])

        # fill in the non-linear part - 2nd order
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                self.out_train[self.dlin+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                cnt += 1
                
        # fill in the non-linear part - 3rd order
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                for span in range(column,self.dlin):
                    self.out_train[self.dlin+self.dnonlin2+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[span,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                    cnt += 1

    # Compute reservoir with 1st & 4th order features
    def reservoir14(self):
        # Compute the features
        # size of nonlinear part of feature vector
        self.dnonlin = int(self.dlin*(self.dlin+1)*(self.dlin+2)*(self.dlin+3)/24)
        # total size of feature vector: bias [possbily] + linear + nonlinear
        self.dtot = int(self.bias) + self.dlin + self.dnonlin      
        
        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        self.out_train = torch.ones((self.dtot,self.system.traintime_pts), dtype=torch.float64, device=self.device)
                
        # copy over the linear part (shift over by one to account for constant)
        self.out_train[int(self.bias):self.dlin+int(self.bias),:]=torch.clone(self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1])
                
        # fill in the non-linear part
        cnt=0
        for row in range(self.dlin):
            for column in range(row,self.dlin):
                for span in range(column,self.dlin):
                    for span4 in range(span,self.dlin):
                        self.out_train[self.dlin+cnt+int(self.bias)]=self.buffer[row,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[column,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[span,self.system.warmup_pts-1:self.system.warmtrain_pts-1]*self.buffer[span4,self.system.warmup_pts-1:self.system.warmtrain_pts-1]
                        cnt += 1

    # Predict in the autoregressive mode
    def predict1(self):
        # Create a place to store feature vectors for prediction
        self.out_test = torch.ones(self.dtot, dtype=torch.float64, device=self.device)              # full feature vector
        self.prediction = torch.zeros((self.dlin,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.buffer[:,self.system.warmtrain_pts-1])
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            # copy linear part into whole feature vector
            self.out_test[int(self.bias):self.dlin+int(self.bias)]=self.prediction[:,j]

            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ self.out_test[:]

            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=torch.clone(self.buffer[0:self.d,self.system.warmtrain_pts+j-tap])
                    
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)    
          
    # Predict in the autoregressive mode
    def predict12(self):
        # Create a place to store feature vectors for prediction
        self.out_test = torch.ones(self.dtot, dtype=torch.float64, device=self.device)              # full feature vector
        self.prediction = torch.zeros((self.dlin,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.buffer[:,self.system.warmtrain_pts-1])
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            # copy linear part into whole feature vector
            self.out_test[int(self.bias):self.dlin+int(self.bias)]=self.prediction[:,j]
            self.feature_2nd=self.prediction[:,j:j+1]@self.prediction[:,j:j+1].T
            self.out_test[self.dlin+int(self.bias):] = self.feature_2nd[self.indexing_2nd]

            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ self.out_test[:]

            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=torch.clone(self.buffer[0:self.d,self.system.warmtrain_pts+j-tap])
                    
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      
        
    def predict13(self):
        # Create a place to store feature vectors for prediction
        self.out_test = torch.ones(self.dtot, dtype=torch.float64, device=self.device)              # full feature vector
        self.prediction = torch.zeros((self.dlin,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.buffer[:,self.system.warmtrain_pts-1])
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            #t = time.time()
            # copy linear part into whole feature vector
            self.out_test[int(self.bias):self.dlin+int(self.bias)]=self.prediction[:,j]
            # fill in the non-linear part
            self.feature_3rd=torch.matmul(torch.unsqueeze(self.prediction[:,j:j+1]@self.prediction[:,j:j+1].T,2), torch.unsqueeze(self.prediction[:,j:j+1].T, 0))
            self.out_test[self.dlin+int(self.bias):] = self.feature_3rd[self.indexing_3rd]
            
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ self.out_test[:]
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=torch.clone(self.buffer[0:self.d,self.system.warmtrain_pts+j-tap])
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      

    def predict14(self):
        # Create a place to store feature vectors for prediction
        self.out_test = torch.ones(self.dtot, dtype=torch.float64, device=self.device)              # full feature vector
        self.prediction = torch.zeros((self.dlin,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.buffer[:,self.system.warmtrain_pts-1])
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            self.out_test[int(self.bias):self.dlin+int(self.bias)]=self.prediction[:,j]
            # fill in the non-linear part                        
            self.feature_4th=torch.matmul(torch.unsqueeze(torch.matmul(torch.unsqueeze(self.prediction[:,j:j+1]@self.prediction[:,j:j+1].T,2), torch.unsqueeze(self.prediction[:,j:j+1].T, 0)),3), torch.unsqueeze(torch.unsqueeze(self.prediction[:,j:j+1].T, 0), 0))            
            self.out_test[self.dlin+int(self.bias):] = self.feature_4th[self.indexing_4th]
            
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ self.out_test[:]
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=torch.clone(self.buffer[0:self.d,self.system.warmtrain_pts+j-tap])
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)  
        
    def predict123(self):
        # Create a place to store feature vectors for prediction
        self.out_test = torch.ones(self.dtot, dtype=torch.float64, device=self.device)              # full feature vector
        self.prediction = torch.zeros((self.dlin,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.buffer[:,self.system.warmtrain_pts-1])
        
        # do prediction
        for j in range(self.system.testtime_pts-1):
            # copy linear part into whole feature vector
            self.out_test[int(self.bias):self.dlin+int(self.bias)]=self.prediction[:,j] 
            # fill in the non-linear part
            self.feature_2nd=self.prediction[:,j:j+1]@self.prediction[:,j:j+1].T
            self.out_test[self.dlin+int(self.bias):self.dlin+int(self.bias)+self.dnonlin2] = self.feature_2nd[self.indexing_2nd]
            self.feature_3rd=torch.matmul(torch.unsqueeze(self.prediction[:,j:j+1]@self.prediction[:,j:j+1].T,2), torch.unsqueeze(self.prediction[:,j:j+1].T, 0))
            self.out_test[self.dlin+int(self.bias)+self.dnonlin2:] = self.feature_3rd[self.indexing_3rd]

            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ self.out_test[:]
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=torch.clone(self.buffer[0:self.d,self.system.warmtrain_pts+j-tap])
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)    
