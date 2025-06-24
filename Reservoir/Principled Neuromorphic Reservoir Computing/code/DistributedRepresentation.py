# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from scipy import sparse
import torch


class DistributedRepresentationRC():

    def __init__(
        self,
        system,
        D,
        k = 2,
        taps = None,
        ridge_param = 0,
        bias = True,
        normalize = True,
        seed = 0,
    ):
        super(DistributedRepresentationRC, self).__init__()
        np.random.seed(seed)
        self.system = system
        self.D = D
        self.D_nb = D - int(bias)
        # number of time delay taps
        self.k = k
        self.ridge_param = ridge_param
        self.bias = bias
        self.normalize = normalize
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
        # create an array to hold the linear part of the feature space
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
        self.out_train_hd=self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1]

        # create necessary vectors 
        self.pos=np.random.randn(self.dlin, self.D_nb)
        self.pos=np.fft.fft(self.pos, axis=1)/np.abs(np.fft.fft(self.pos, axis=1))        
        self.B = np.fft.ifft(self.pos, axis=1).real
        self.perm = np.random.permutation(self.D_nb)

        if self.normalize:
            self.std_train = np.std(self.out_train_hd[0:self.d,:], axis=1, keepdims=True)
            self.std_train = np.expand_dims(np.tile(self.std_train,(self.k)).flatten(order='F'), axis=1)            
            self.out_train_hd=self.out_train_hd/self.std_train        
                
        self.reservoir_states = np.ones((self.D,self.system.traintime_pts))
        for i in range(self.system.traintime_pts):
            OlinHD=np.sum(self.out_train_hd[:,i:i+1]*self.B, axis=0, keepdims=True).real
            OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])).real
            
            # Joint representation
            self.reservoir_states[int(self.bias):,i:i+1] = np.transpose(1.*OnlHD+1.*OlinHD)
            
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]) @ self.reservoir_states[:,:].T @ np.linalg.pinv(self.reservoir_states[:,:] @ self.reservoir_states[:,:].T +self.ridge_param*np.identity(self.D))

        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.reservoir_states[:,0:self.system.traintime_pts]

        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()
        
    # Train the classfier for 1- & 3- order features
    def fit13(self):
        # Compute the features
        self.out_train_hd=self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1]

        # create necessary vectors 
        self.perm = np.random.permutation(self.D_nb)
        self.perm2 =  self.perm[self.perm]
        self.pos=np.random.randn(self.dlin, self.D_nb)
        self.pos=np.fft.fft(self.pos, axis=1)/np.abs(np.fft.fft(self.pos, axis=1))        
        self.B = np.fft.ifft(self.pos, axis=1).real

        if self.normalize:
            self.std_train = np.std(self.out_train_hd[0:self.d,:], axis=1, keepdims=True)
            self.std_train = np.expand_dims(np.tile(self.std_train,(self.k)).flatten(order='F'), axis=1)            
            self.out_train_hd=self.out_train_hd/self.std_train        
                
        self.reservoir_states = np.ones((self.D,self.system.traintime_pts))
        for i in range(self.system.traintime_pts):
            OlinHD=np.sum(self.out_train_hd[:,i:i+1]*self.B, axis=0, keepdims=True).real
            OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])*np.fft.fft(OlinHD[:,self.perm2])).real
            
            # Joint representation
            self.reservoir_states[int(self.bias):,i:i+1] = np.transpose(1.*OnlHD+1.*OlinHD)
            
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]) @ self.reservoir_states[:,:].T @ np.linalg.pinv(self.reservoir_states[:,:] @ self.reservoir_states[:,:].T +self.ridge_param*np.identity(self.D))

        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.reservoir_states[:,0:self.system.traintime_pts]

        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()

    # Train the classfier for 1- 2- 3- & 4- order features
    def fit1234(self):
        # Compute the features
        self.out_train_hd=self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1]

        # create necessary vectors 
        self.pos=np.random.randn(self.dlin, self.D_nb)
        self.pos=np.fft.fft(self.pos, axis=1)/np.abs(np.fft.fft(self.pos, axis=1))        
        self.B = np.fft.ifft(self.pos, axis=1).real
        self.perm = np.random.permutation(self.D_nb)
        self.perm2 =  self.perm[self.perm]
        self.perm3 =  self.perm2[self.perm]

        if self.normalize:
            self.std_train = np.std(self.out_train_hd[0:self.d,:], axis=1, keepdims=True)
            self.std_train = np.expand_dims(np.tile(self.std_train,(self.k)).flatten(order='F'), axis=1)            
            self.out_train_hd=self.out_train_hd/self.std_train        
                
        self.reservoir_states = np.ones((self.D,self.system.traintime_pts))
        for i in range(self.system.traintime_pts):
            OlinHD=np.sum(self.out_train_hd[:,i:i+1]*self.B, axis=0, keepdims=True).real            
            OnlHD2 = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])).real
            OnlHD3 = np.fft.ifft (np.fft.fft(OnlHD2)*np.fft.fft(OlinHD[:,self.perm2])).real
            OnlHD4 = np.fft.ifft (np.fft.fft(OnlHD3)*np.fft.fft(OlinHD[:,self.perm3])).real

            # Joint representation
            self.reservoir_states[int(self.bias):,i:i+1] = np.transpose(OnlHD4+OnlHD3+OnlHD2+OlinHD)
            
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]) @ self.reservoir_states[:,:].T @ np.linalg.pinv(self.reservoir_states[:,:] @ self.reservoir_states[:,:].T +self.ridge_param*np.identity(self.D))

        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.reservoir_states[:,0:self.system.traintime_pts]

        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()

    # Perform the prediction
    def predict12(self):      
        # create a place to store feature vectors for prediction
        self.prediction = np.zeros((self.dlin,self.system.testtime_pts)) # linear part
        # copy over initial linear feature vector
        self.prediction[:,0] = self.buffer[:,self.system.warmtrain_pts-1]       

        # do prediction
        for j in range(self.system.testtime_pts-1):
            inp = self.prediction[:,j:j+1]
            if self.normalize:
                inp = inp/self.std_train
        
            OlinHD=np.sum(inp*self.B, axis=0, keepdims=True).real
            OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])).real
            
            # Joint representation
            Otot = np.ones((self.D,1))
            Otot[int(self.bias):,0:1] = np.transpose(1.*OnlHD+1.*OlinHD)    
                           
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ Otot[:,0]
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      
        
    # Perform the prediction
    def predict13(self):      
        # create a place to store feature vectors for prediction
        self.prediction = np.zeros((self.dlin,self.system.testtime_pts)) # linear part
        # copy over initial linear feature vector
        self.prediction[:,0] = self.buffer[:,self.system.warmtrain_pts-1]       

        # do prediction
        for j in range(self.system.testtime_pts-1):
            inp = self.prediction[:,j:j+1]
            if self.normalize:
                inp = inp/self.std_train
                
            OlinHD=np.sum(inp*self.B, axis=0, keepdims=True).real
            OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])*np.fft.fft(OlinHD[:,self.perm2])).real
            
            # Joint representation
            Otot = np.ones((self.D,1))
            Otot[int(self.bias):,0:1] = np.transpose(1.*OnlHD+1.*OlinHD)    
                          
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ Otot[:,0]
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      

    # Perform the prediction
    def predict1234(self):      
        # create a place to store feature vectors for prediction
        self.prediction = np.zeros((self.dlin,self.system.testtime_pts)) # linear part
        # copy over initial linear feature vector
        self.prediction[:,0] = self.buffer[:,self.system.warmtrain_pts-1]       

        # do prediction
        for j in range(self.system.testtime_pts-1):
            inp = self.prediction[:,j:j+1]
            if self.normalize:
                inp = inp/self.std_train
        
            OlinHD=np.sum(inp*self.B, axis=0, keepdims=True).real
            OnlHD2 = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])).real
            OnlHD3 = np.fft.ifft (np.fft.fft(OnlHD2)*np.fft.fft(OlinHD[:,self.perm2])).real
            OnlHD4 = np.fft.ifft (np.fft.fft(OnlHD3)*np.fft.fft(OlinHD[:,self.perm3])).real
            
            Otot = np.ones((self.D,1))
            Otot[int(self.bias):,0:1] = np.transpose(OnlHD4+OnlHD3+OnlHD2+OlinHD)   
                           
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ Otot[:,0]
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)     



class DistributedRepresentationRCTorch():

    def __init__(
        self,
        system,
        device,
        D,
        k = 2,
        taps = None,
        ridge_param = 0,
        bias = True,
        normalize = True,
        seed = 0,
    ):
        super(DistributedRepresentationRCTorch, self).__init__()
        np.random.seed(seed)
        self.system = system
        self.device = device
        self.D = D
        self.D_nb = D - int(bias)
        # number of time delay taps
        self.k = k
        self.ridge_param = ridge_param
        self.bias = bias
        self.normalize = normalize
        # input dimension
        self.d=self.system.data.y.shape[0]        
        # size of linear part of feature vector
        self.dlin = self.k*self.d
        if taps is None:
            self.taps = torch.arange(0,k)
        else:
            self.taps = taps        
        self.memoryBuffer()

        # create necessary vectors 
        self.pos=np.random.randn(self.dlin, self.D_nb)
        self.pos=np.fft.fft(self.pos, axis=1)/np.abs(np.fft.fft(self.pos, axis=1))        
        self.B = torch.from_numpy(np.fft.ifft(self.pos, axis=1).real).to(self.device)
        self.perm = torch.from_numpy(np.random.permutation(self.D_nb)).to(self.device)
        self.perm2 =  self.perm[self.perm]
        self.perm3 =  self.perm2[self.perm]

        # Compute the features
        self.out_train_hd=torch.clone(self.buffer[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1])

        if self.normalize:
            self.std_train = torch.std(self.out_train_hd[0:self.d,:], dim=1, keepdim=True)            
            self.std_train = torch.tile(self.std_train, (self.k,1))            
            self.out_train_hd=self.out_train_hd/self.std_train        
        
        add_noise = True

    # Compute readout
    def readout(self):
        
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = torch.linalg.lstsq(self.reservoir_states[:,:] @ self.reservoir_states[:,:].T +self.ridge_param*torch.eye(self.D, dtype=torch.float64, device=self.device), self.reservoir_states[:,:] @  (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]).T, driver='gels').solution.T
        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.reservoir_states[:,0:self.system.traintime_pts]


    # Compute readout
    def readoutInv(self):
        
        # Compute the classifier
        # Ridge regression: train W_out to map out_train to the difference of the system states
        self.W_out = (torch.linalg.lstsq(self.reservoir_states[:,:].T @ self.reservoir_states[:,:] +self.ridge_param*torch.eye(self.system.traintime_pts, dtype=torch.float64, device=self.device),  (self.buffer[0:self.d,self.system.warmup_pts:self.system.warmtrain_pts]-self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1]).T).solution.T)@self.reservoir_states[:,0:self.system.traintime_pts].T
        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.buffer[0:self.d,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.reservoir_states[:,0:self.system.traintime_pts]

        
    # Compute reservoir with 1st & 2nd order features
    def reservoir13(self):
        self.reservoir_states = torch.ones((self.D,self.system.traintime_pts), dtype=torch.float64, device=self.device)
        for i in range(self.system.traintime_pts):
            OlinHD= torch.sum(self.out_train_hd[:,i:i+1]*self.B, dim=0, keepdims=True)
            OnlHD = torch.fft.ifft (torch.fft.fft(OlinHD)*torch.fft.fft(OlinHD[:,self.perm])*torch.fft.fft(OlinHD[:,self.perm2])).real
            
            # Joint representation
            self.reservoir_states[int(self.bias):,i:i+1] = ((1./3)*OnlHD+1.*OlinHD).T


    # Compute reservoir with 1st & 2nd order features
    def reservoir123(self):
        self.reservoir_states = torch.ones((self.D,self.system.traintime_pts), dtype=torch.float64, device=self.device)
        for i in range(self.system.traintime_pts):
            OlinHD= (1/np.sqrt(self.dlin))*torch.sum(self.out_train_hd[:,i:i+1]*self.B, dim=0, keepdims=True)
            OnlHD2 = torch.fft.ifft (torch.fft.fft(OlinHD)*torch.fft.fft(OlinHD[:,self.perm])).real
            OnlHD3 = torch.fft.ifft (torch.fft.fft(OnlHD2)*torch.fft.fft(OlinHD[:,self.perm2])).real
            
            #self.reservoir_states[int(self.bias):,i:i+1] = ((1./3)*OnlHD3+(1./2)*OnlHD2+1.*OlinHD).T
            self.reservoir_states[int(self.bias):,i:i+1] = ((1)*OnlHD3+(1)*OnlHD2+1.*OlinHD).T

    # Train the classfier for 1- 2- 3- & 4- order features
    def reservoir1234(self):
        self.reservoir_states = torch.ones((self.D,self.system.traintime_pts), dtype=torch.float64, device=self.device)
        for i in range(self.system.traintime_pts):
            OlinHD= (1/np.sqrt(self.dlin))*torch.sum(self.out_train_hd[:,i:i+1]*self.B, dim=0, keepdims=True)
            OnlHD2 = torch.fft.ifft (torch.fft.fft(OlinHD)*torch.fft.fft(OlinHD[:,self.perm])).real
            OnlHD3 = torch.fft.ifft (torch.fft.fft(OnlHD2)*torch.fft.fft(OlinHD[:,self.perm2])).real
            OnlHD4 = torch.fft.ifft (torch.fft.fft(OnlHD3)*torch.fft.fft(OlinHD[:,self.perm3])).real
            
            #self.reservoir_states[int(self.bias):,i:i+1] = ((1./3)*OnlHD3+(1./2)*OnlHD2+1.*OlinHD).T
            self.reservoir_states[int(self.bias):,i:i+1] = ((1)*OnlHD4+(1)*OnlHD3+(1)*OnlHD2+1.*OlinHD).T

    # Form the memory buffer
    def memoryBuffer(self):
        # create an array to hold the linear part of the feature space
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
        
    # Train the classfier for 1- & 2- order features
    def fit12(self):
                
        self.reservoir_states = np.ones((self.D,self.system.traintime_pts))
        for i in range(self.system.traintime_pts):
            OlinHD=np.sum(self.out_train_hd[:,i:i+1]*self.B, axis=0, keepdims=True).real
            OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])).real
            # Joint representation
            self.reservoir_states[int(self.bias):,i:i+1] = np.transpose(1.*OnlHD+1.*OlinHD)                
        self.readout()
            
        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()
        
    # Train the classfier for 1- & 3- order features
    def fit13(self):
                
        self.reservoir_states = np.ones((self.D,self.system.traintime_pts))
        for i in range(self.system.traintime_pts):
            OlinHD=np.sum(self.out_train_hd[:,i:i+1]*self.B, axis=0, keepdims=True).real
            OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])*np.fft.fft(OlinHD[:,self.perm2])).real
            
            # Joint representation
            self.reservoir_states[int(self.bias):,i:i+1] = np.transpose(1.*OnlHD+1.*OlinHD)
 
        self.readout()            
 
        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()

    # Train the classfier for 1- 2- 3- & 4- order features
    def fit1234(self):
        self.reservoir_states = np.ones((self.D,self.system.traintime_pts))
        for i in range(self.system.traintime_pts):
            OlinHD=np.sum(self.out_train_hd[:,i:i+1]*self.B, axis=0, keepdims=True).real            
            OnlHD2 = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])).real
            OnlHD3 = np.fft.ifft (np.fft.fft(OnlHD2)*np.fft.fft(OlinHD[:,self.perm2])).real
            OnlHD4 = np.fft.ifft (np.fft.fft(OnlHD3)*np.fft.fft(OlinHD[:,self.perm3])).real

            # Joint representation
            self.reservoir_states[int(self.bias):,i:i+1] = np.transpose(OnlHD4+OnlHD3+OnlHD2+OlinHD)
        
        self.readout()    

        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()

    # Perform the prediction
    def predict12(self):      
        # create a place to store feature vectors for prediction
        self.prediction = np.zeros((self.dlin,self.system.testtime_pts)) # linear part
        # copy over initial linear feature vector
        self.prediction[:,0] = self.buffer[:,self.system.warmtrain_pts-1]       

        # do prediction
        for j in range(self.system.testtime_pts-1):
            inp = self.prediction[:,j:j+1]
            if self.normalize:
                inp = inp/self.std_train
        
            OlinHD=np.sum(inp*self.B, axis=0, keepdims=True).real
            OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,self.perm])).real
            
            # Joint representation
            Otot = np.ones((self.D,1))
            Otot[int(self.bias):,0:1] = np.transpose(1.*OnlHD+1.*OlinHD)    
                           
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ Otot[:,0]
            
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      
        
    # Perform the prediction
    def predict13(self):      
        # create a place to store feature vectors for prediction
        self.prediction = torch.zeros((self.dlin,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.buffer[:,self.system.warmtrain_pts-1])       

        # do prediction
        for j in range(self.system.testtime_pts-1):
            inp = torch.clone(self.prediction[:,j:j+1])
            if self.normalize:
                inp = inp/self.std_train
                
            OlinHD=torch.sum(inp*self.B, dim=0, keepdims=True)
            OnlHD = torch.fft.ifft (torch.fft.fft(OlinHD)*torch.fft.fft(OlinHD[:,self.perm])*torch.fft.fft(OlinHD[:,self.perm2])).real
            
            # Joint representation
            Otot = torch.ones((self.D,1), dtype=torch.float64, device=self.device)
            Otot[int(self.bias):,0:1] = ((1./3)*OnlHD+1.*OlinHD).T    
                          
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ Otot[:,0]
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)      

    # Perform the prediction
    def predict123(self):      
        # create a place to store feature vectors for prediction
        self.prediction = torch.zeros((self.dlin,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.buffer[:,self.system.warmtrain_pts-1])       

        # do prediction
        for j in range(self.system.testtime_pts-1):
            inp = torch.clone(self.prediction[:,j:j+1])
            if self.normalize:
                inp = inp/self.std_train
                
            OlinHD = (1/np.sqrt(self.dlin))*torch.sum(inp*self.B, dim=0, keepdims=True)
            OnlHD2 = torch.fft.ifft (torch.fft.fft(OlinHD)*torch.fft.fft(OlinHD[:,self.perm])).real
            OnlHD3 = torch.fft.ifft (torch.fft.fft(OnlHD2)*torch.fft.fft(OlinHD[:,self.perm2])).real            
                        
            # Joint representation
            Otot = torch.ones((self.D,1), dtype=torch.float64, device=self.device)
            #Otot[int(self.bias):,0:1] = ((1./3)*OnlHD3+(1./2)*OnlHD2+1.*OlinHD).T    
            Otot[int(self.bias):,0:1] = ((1.)*OnlHD3+(1.)*OnlHD2+1.*OlinHD).T    
                          
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ Otot[:,0]
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)    

    # Perform the prediction
    def predict1234(self):      
        # create a place to store feature vectors for prediction
        self.prediction = torch.zeros((self.dlin,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.buffer[:,self.system.warmtrain_pts-1])         

        # do prediction
        for j in range(self.system.testtime_pts-1):
            inp = torch.clone(self.prediction[:,j:j+1])
            if self.normalize:
                inp = inp/self.std_train
                
            OlinHD = (1/np.sqrt(self.dlin))*torch.sum(inp*self.B, dim=0, keepdims=True)
            OnlHD2 = torch.fft.ifft (torch.fft.fft(OlinHD)*torch.fft.fft(OlinHD[:,self.perm])).real
            OnlHD3 = torch.fft.ifft (torch.fft.fft(OnlHD2)*torch.fft.fft(OlinHD[:,self.perm2])).real    
            OnlHD4 = torch.fft.ifft (torch.fft.fft(OnlHD3)*torch.fft.fft(OlinHD[:,self.perm3])).real   

            # Joint representation
            Otot = torch.ones((self.D,1), dtype=torch.float64, device=self.device)
            #Otot[int(self.bias):,0:1] = ((1./3)*OnlHD3+(1./2)*OnlHD2+1.*OlinHD).T    
            Otot[int(self.bias):,0:1] = ((1.)*OnlHD4+(1.)*OnlHD3+(1.)*OnlHD2+1.*OlinHD).T  
                                       
            # do a prediction
            self.prediction[0:self.d,j+1] = self.prediction[0:self.d,j]+self.W_out @ Otot[:,0]
            # fill in the delay taps of the next state
            for delay in range(1,self.k):
                tap = self.taps[delay]
                if j+1>=tap:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.prediction[0:self.d,j+1-tap]
                else:
                    self.prediction[delay*self.d:(delay+1)*self.d,j+1]=self.buffer[0:self.d,self.system.warmtrain_pts+j-tap]
        
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)     