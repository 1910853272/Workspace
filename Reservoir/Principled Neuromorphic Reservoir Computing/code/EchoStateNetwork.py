# -*- coding: utf-8 -*-
"""

"""

import math
import numpy as np
import torch

class EchoStateNetworkRCTorch():
    """Class for Echo State Networks
    """

    def __init__(
        self,
        system,
        device,
        D: int,
        input_scale: float = 1.0,
        spectral_radius: float = 1.0,
        include_bias: bool = True,
        normalize: bool = False,
        ridge_param: float = 0.0,
        seed_id: int = 0,      
    ):
        self.system = system
        self.device = device
        self.D = D
        self.input_scale = input_scale
        self.spectral_radius = spectral_radius
        self.include_bias = include_bias
        self.normalize = normalize
        self.ridge_param = ridge_param
        #the dimensionality of the input time-series.
        self.d=self.system.data.y.shape[0]        
        self.reservoir_extension = 0

        # Check if bias is added into the input
        if self.include_bias:
            self.in_channels_bias = self.d + 1 
        else:
            self.in_channels_bias = self.d
                   
        ##
        ## Initialize ESN parameters based on randomness
        ##        
        
        # Set seed
        np.random.seed(seed=seed_id)
        
        if torch.cuda.is_available():
            gen = torch.Generator(device='cuda').manual_seed(seed_id)      
        else:
            gen = torch.Generator(device='cpu').manual_seed(seed_id)                  
        
        #Uniform random projection matrix for feeding input into the reservoir  
        #Scaling Win by input_scale
        self.Win = torch.zeros((self.D, self.in_channels_bias), dtype=torch.float64, device=self.device) 
        self.Win.uniform_(generator=gen)  
        self.Win = self.input_scale*2*(self.Win-0.5)
       
        #Recurrent connectivity matrix W for the reservoir
        #Assign random values to these connections
        self.W = torch.zeros((self.D, self.D), dtype=torch.float64, device=self.device)        
        self.W.normal_(generator=gen)
        
        # Scale the resuling recurrent connectivity matrix 
        w = torch.linalg.eigvals(self.W)
        self.W = self.spectral_radius*self.W/torch.abs(w[0]) 

    # Compute NRMSE
    def nrmse(self, train = True):
        if train:
            nrmse_value = torch.sqrt(torch.mean((self.system.data.y[:,self.system.warmup_pts:self.system.warmtrain_pts]-self.prediction_train[:,:])**2)/self.system.total_var)    
        else:
            nrmse_value = torch.sqrt(torch.mean((self.system.data.y[:,self.system.warmtrain_pts-1:self.system.warmtrain_pts+self.system.lyaptime_pts-1]-self.prediction[:,0:self.system.lyaptime_pts])**2)/self.system.total_var)
        return nrmse_value 

    # Teacher forcing
    def reservoir_train(self):
        # Prepare the training data
        self.training_data = torch.clone(self.system.data.y[:,0:self.system.warmtrain_pts-1]) 
        
        if self.normalize:
            self.std_train = torch.std(self.training_data[:,:], dim=1, keepdims=True)
            self.training_data=self.training_data/self.std_train           
        
        if self.include_bias:
            self.training_data = torch.concatenate((1*torch.ones((1, self.system.warmtrain_pts-1), dtype=torch.float64, device=self.device), self.training_data, ), dim=0)        
        
        # Initialize reservoir's state
        self.reservoir = torch.zeros((self.D+self.reservoir_extension, 1), dtype=torch.float64, device=self.device)

        # Obtain the reservoir states for the training phase
        self.reservoir_states = torch.zeros((self.D+self.reservoir_extension, self.system.warmtrain_pts-1), dtype=torch.float64, device=self.device)
        for i in range(0,self.system.warmtrain_pts-1):    
            # Update the reservoir
            self.reservoir[self.reservoir_extension:,0:1] = torch.tanh(self.Win @ self.training_data[:,i:i+1] + self.W @ self.reservoir[self.reservoir_extension:,0:1])
            self.reservoir_states[:,i:i+1] = torch.clone(self.reservoir)
        self.reservoir_train_last = torch.clone(self.reservoir)
        

    # Compute the readout
    def readout(self):
        # Ridge regression: train W_out to map out_train to the difference of the system states
        try:
            self.W_out = torch.linalg.lstsq(self.reservoir_states[:,self.system.warmup_pts-1:] @ self.reservoir_states[:,self.system.warmup_pts-1:].T +self.ridge_param*torch.eye(self.D+self.reservoir_extension, dtype=torch.float64, device=self.device), self.reservoir_states[:,self.system.warmup_pts-1:] @  (self.system.data.y[:,self.system.warmup_pts:self.system.warmtrain_pts]-self.system.data.y[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1]).T, driver='gels').solution.T
        except:
            print("The readout matrix was not computed due to some numerical errors")
            self.W_out = torch.zeros((self.system.data.y.shape[0],self.D+self.reservoir_extension), dtype=torch.float64, device=self.device)

        # Apply W_out to the training feature vector to get the training output
        self.prediction_train = self.system.data.y[:,self.system.warmup_pts-1:self.system.warmtrain_pts-1] + self.W_out @ self.reservoir_states[:,self.system.warmup_pts-1:]
             
    # Performs the training phase of the Echo State Network.
    def fit(self):        
        self.reservoir_train()
        self.readout()        
        # Calculate NRMSE between ground truth and training output
        self.nrmse_train = self.nrmse()  
        
    # Perform the prediction
    def predict(self):      
        # create a place to store feature vectors for prediction
        self.prediction = torch.zeros((self.d,self.system.testtime_pts), dtype=torch.float64, device=self.device) # linear part
        # copy over initial linear feature vector
        self.prediction[:,0] = torch.clone(self.system.data.y[:,self.system.warmtrain_pts-1])       

        self.reservoir = torch.clone(self.reservoir_train_last)
        # do prediction
        for j in range(self.system.testtime_pts-1):
            inp = torch.clone(self.prediction[:,j:j+1])
            if self.normalize:
                inp = inp/self.std_train
            if self.include_bias:
                inp = torch.concatenate((1*torch.ones((1, 1), dtype=torch.float64, device=self.device), inp,), dim=0)            
            # Update the reservoir
            self.reservoir[self.reservoir_extension:,0:1] = torch.tanh(self.Win@ inp + self.W @ self.reservoir[self.reservoir_extension:,0:1])         
            
            self.prediction[:,j+1] = self.prediction[:,j]+self.W_out @ self.reservoir[:,0]
               
        # calculate NRMSE between true signal and prediction for the chosen number of Lyapunov times
        self.nrmse_test = self.nrmse(train=False)   


