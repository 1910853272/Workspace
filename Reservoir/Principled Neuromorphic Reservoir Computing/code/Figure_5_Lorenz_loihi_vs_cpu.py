# -*- coding: utf-8 -*-
"""

This script plots the results of experiments reported in panels b & c of Figure 5.

"""

import numpy as np
from scipy import stats
import sklearn.preprocessing
import matplotlib.pyplot as plt
from DataGeneration import Lorenz63

# Specify parameters of the model
ridge_param = 1.e-3
N=200
block_size=20
block_num = int(N/block_size)
decay_step = 0.5
train_samples = 700
use_bias = True # [False | True]
# Files for the results of experiments for three different variants of RC on Loihi2
files = ['/data/Figure_5_Loihi_N=200_K=20_run_1.npz',
        '/data/Figure_5_Loihi_N=200_K=20_run_2.npz',
        '/data/Figure_5_Loihi_N=200_K=20_run_3.npz',
        '/data/Figure_5_Loihi_N=200_K=20_run_4.npz',
        '/data/Figure_5_Loihi_N=200_K=20_run_5.npz',
        '/data/Figure_5_Loihi_N=200_K=20_run_6.npz',
        '/data/Figure_5_Loihi_N=200_K=20_run_7.npz',
        '/data/Figure_5_Loihi_N=200_K=20_run_8.npz']
simul = len(files)

# Variables for collecting the statistics for three different variants of RC
nrmse_buffer_blocks=np.zeros((block_num, simul))
nrmse_single_1st_2nd_blocks=np.zeros((block_num, simul))
nrmse_buffer_2nd_order_blocks=np.zeros((block_num, simul))

## Process Loihi2 results ##
for i_s in range(0,simul):  
    # Load data
    data = np.load(files[i_s])
    
    # Offsets to account for time-delayed processing of Loihi2
    offset_buffer = data['f_offset']
    offset_sigmapi = data['offset']
    
    # Distributed representations computed by Loihi2
    #1st-order features in the memory buffer
    spike_data_buffer =  data['f_out_spike_data'][:, offset_buffer:]
    spike_data_buffer = np.concatenate((np.ones((1,spike_data_buffer.shape[1])), spike_data_buffer), axis=0)
    #1st- & 2nd- order features from the most recent time step
    spike_data_single_1st_2nd_order = data['out_spike_data_0buf'][:, offset_sigmapi:]
    spike_data_single_1st_2nd_order = np.concatenate((np.ones((1,spike_data_single_1st_2nd_order.shape[1])), spike_data_single_1st_2nd_order), axis=0)
    #1st- & 2nd- order features from the memory buffer 
    spike_data_buffer_2nd_order = data['out_spike_data'][:, offset_sigmapi:]
    spike_data_buffer_2nd_order = np.concatenate((np.ones((1,spike_data_buffer_2nd_order.shape[1])), spike_data_buffer_2nd_order), axis=0) 

    #Target functions with proper offset
    target = data['target']
    #Target for memory buffer
    target_buffer = target[:, :-offset_buffer]
    target_buffer_var = np.var(target_buffer)
    #Target for variants involving 2nd-order features
    target_sigmapi = target[:, :- offset_sigmapi]        
    target_sigmapi_var = np.var(target_sigmapi)
       
    # Process blocks sequentially
    for i_b in range(block_num):    
        # Compute the readout matrix for each variant and get the predictions          
        Wout_buffer = (target_buffer[:,0:train_samples] @ spike_data_buffer[0:(i_b+1)*block_size+1,0:train_samples].T 
                          @ np.linalg.pinv(spike_data_buffer[0:(i_b+1)*block_size+1,0:train_samples] 
                                           @ spike_data_buffer[0:(i_b+1)*block_size+1,0:train_samples].T 
                                           + ridge_param*np.identity((i_b+1)*block_size+1)))
        pred_buffer = Wout_buffer@spike_data_buffer[0:(i_b+1)*block_size+1,train_samples:]         

        Wout_single_1st_2nd = (target_sigmapi[:,0:train_samples] @ spike_data_single_1st_2nd_order[0:(i_b+1)*block_size+1,0:train_samples].T 
                           @ np.linalg.pinv(spike_data_single_1st_2nd_order[0:(i_b+1)*block_size+1,0:train_samples] 
                                            @ spike_data_single_1st_2nd_order[0:(i_b+1)*block_size+1,0:train_samples].T 
                                            + ridge_param*np.identity((i_b+1)*block_size+1)))        
        pred_single_1st_2nd = Wout_single_1st_2nd@spike_data_single_1st_2nd_order[0:(i_b+1)*block_size+1,train_samples:]              
    
        Wout_buffer_2nd_order = (target_sigmapi[:,0:train_samples] @ spike_data_buffer_2nd_order[0:(i_b+1)*block_size+1, 0:train_samples].T 
                          @ np.linalg.pinv(spike_data_buffer_2nd_order[0:(i_b+1)*block_size+1, 0:train_samples] 
                                           @ spike_data_buffer_2nd_order[0:(i_b+1)*block_size+1, 0:train_samples].T 
                                           + ridge_param*np.identity((i_b+1)*block_size+1)))
        pred_buffer_2nd_order = Wout_buffer_2nd_order@spike_data_buffer_2nd_order[0:(i_b+1)*block_size+1,train_samples:]    

        # Compute NRMSE between true target values and predictions for the test data
        nrmse_buffer_blocks[i_b,i_s] = np.sqrt(np.mean((target_buffer[:,train_samples:] - pred_buffer)**2,axis=1)/target_buffer_var)[0]
        nrmse_single_1st_2nd_blocks[i_b,i_s] = np.sqrt(np.mean((target_sigmapi[:,train_samples:] - pred_single_1st_2nd)**2,axis=1)/target_sigmapi_var)[0]        
        nrmse_buffer_2nd_order_blocks[i_b,i_s] = np.sqrt(np.mean((target_sigmapi[:,train_samples:] - pred_buffer_2nd_order)**2,axis=1)/target_sigmapi_var)[0]

# Compute mean NRMSE and its standard errors for all simulations and RC variants
nrmse_buffer_blocks_m = np.mean(nrmse_buffer_blocks, axis=1)
nrmse_single_1st_2nd_blocks_m = np.mean(nrmse_single_1st_2nd_blocks, axis=1)
nrmse_buffer_2nd_order_blocks_m = np.mean(nrmse_buffer_2nd_order_blocks, axis=1)

# Standard errors
nrmse_buffer_blocks_low  = nrmse_buffer_blocks_m - np.std(nrmse_buffer_blocks, axis=1)/np.sqrt(simul)
nrmse_buffer_blocks_high  = nrmse_buffer_blocks_m + np.std(nrmse_buffer_blocks, axis=1)/np.sqrt(simul)

nrmse_single_1st_2nd_blocks_low  = nrmse_single_1st_2nd_blocks_m - np.std(nrmse_single_1st_2nd_blocks, axis=1)/np.sqrt(simul)
nrmse_single_1st_2nd_blocks_high  = nrmse_single_1st_2nd_blocks_m + np.std(nrmse_single_1st_2nd_blocks, axis=1)/np.sqrt(simul)

nrmse_buffer_2nd_order_blocks_low  = nrmse_buffer_2nd_order_blocks_m - np.std(nrmse_buffer_2nd_order_blocks, axis=1)/np.sqrt(simul)
nrmse_buffer_2nd_order_blocks_high  = nrmse_buffer_2nd_order_blocks_m + np.std(nrmse_buffer_2nd_order_blocks, axis=1)/np.sqrt(simul)

# Compute preductions for the last run using all 10 blocks
pred_buffer_loihi_all_blocks = Wout_buffer@spike_data_buffer
pred_single_1st_2nd_loihi_all_blocks = Wout_single_1st_2nd @ spike_data_single_1st_2nd_order
pred_buffer_2nd_order_loihi_all_blocks = Wout_buffer_2nd_order@spike_data_buffer_2nd_order

## Compute CPU results ##
if use_bias:
    N+=1
    
##
## Data - Lorenz63 system
##
lorenz = Lorenz63(warmup = 0.025,traintime = 17.5,testtime=7.5,)
# Remove the warmup part        
data = lorenz.data.y[:,lorenz.warmup_pts:lorenz.maxtime_pts] 
data = stats.zscore(data, axis=1)
data = np.concatenate((np.ones((1,data.shape[1])), data), axis=0)
maxtime_pts = data.shape[1]

# For forming Sparse Block Codes
encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False).fit(np.arange(0,block_size).reshape(-1,1))

# Variables for collecting the statistics for three different variants of RC
cpu_nrmse_buffer_blocks=np.zeros((block_num, simul))
cpu_nrmse_single_1st_2nd_blocks=np.zeros((block_num, simul))
cpu_nrmse_buffer_2nd_order_blocks=np.zeros((block_num, simul))

# Run for simul random codebooks
for i_s in range(0,simul):  
    np.random.seed(seed=i_s)
    
    # Load target function
    target = np.load(files[i_s])['target']
    target_var=np.var(target)
    
    # Generate the random codebook
    codebook = np.zeros((lorenz.data.y.shape[0]+1, block_num*block_size))    
    for i in range(0,lorenz.data.y.shape[0]+1):  
        num = np.random.randint(0, high=block_size, size=(block_num,1))
        block_code = encoder.transform(num)
        codebook[i,:] =  block_code.flatten()
        
    # Store the computed distributed representations
    buffer = np.ones((N,maxtime_pts))
    single_1st_2nd_order = np.ones((N,maxtime_pts))
    buffer_2nd_order = np.ones((N,maxtime_pts))
    
    reservoir_buffer = np.zeros((block_num,block_size))
    for i in range(maxtime_pts):
        # Random projection step
        buffer_input=np.sum(data[:,i:i+1]*codebook, axis=0, keepdims=True)
        buffer_input=np.reshape(buffer_input,(block_num,block_size))
        
        # Evolve the reservoir
        reservoir_buffer = decay_step * np.roll(reservoir_buffer,1,axis=1) + buffer_input
        reservoir_state = np.expand_dims(reservoir_buffer.flatten(), axis=0) 
             
        #Perform the binding operation via Block-wise FFT
        reservoir_state_fft = np.fft.fft(np.reshape(reservoir_state,(block_num,block_size)), axis=1)        
        second_order_fft = reservoir_state_fft*reservoir_state_fft        
        second_order = np.fft.ifft(second_order_fft, axis=1).real.flatten()
        
        ## Compute features when considering only the most recent time step
        buffer_input_fft = np.fft.fft(buffer_input, axis=1)  
        second_order_1step_fft = buffer_input_fft*buffer_input_fft
        second_order_1step = np.fft.ifft(second_order_1step_fft, axis=1).real.flatten()
        
        # Collect distributed representations for each variant  
        buffer[int(use_bias):,i] = reservoir_state[0,:]        
        single_1st_2nd_order[int(use_bias):,i] = second_order_1step        
        buffer_2nd_order[int(use_bias):,i] = second_order 

    #Readout increasing  the number of blocks
    for i_b in range(block_num):   
        # Compute the readout matrix for each variant and get the predictions
        Wout_buffer = target[:,0:train_samples] @ buffer[0:1+(i_b+1)*block_size,0:train_samples].T @ np.linalg.pinv(buffer[0:1+(i_b+1)*block_size,0:train_samples] @ buffer[0:1+(i_b+1)*block_size,0:train_samples].T + ridge_param*np.identity(1+(i_b+1)*block_size))
        pred_buffer = Wout_buffer@buffer[0:1+(i_b+1)*block_size,train_samples:]    

        Wout_single_1st_2nd = target[:,0:train_samples] @ single_1st_2nd_order[0:1+(i_b+1)*block_size,0:train_samples].T @ np.linalg.pinv(single_1st_2nd_order[0:1+(i_b+1)*block_size,0:train_samples] @ single_1st_2nd_order[0:1+(i_b+1)*block_size,0:train_samples].T + ridge_param*np.identity(1+(i_b+1)*block_size))
        pred_single_1st_2nd = Wout_single_1st_2nd@single_1st_2nd_order[0:1+(i_b+1)*block_size,train_samples:]  

        Wout_buffer_2nd_order = target[:,0:train_samples] @ buffer_2nd_order[0:1+(i_b+1)*block_size,0:train_samples].T @ np.linalg.pinv(buffer_2nd_order[0:1+(i_b+1)*block_size,0:train_samples] @ buffer_2nd_order[0:1+(i_b+1)*block_size,0:train_samples].T + ridge_param*np.identity(1+(i_b+1)*block_size))
        pred_buffer_2nd_order = Wout_buffer_2nd_order@buffer_2nd_order[0:1+(i_b+1)*block_size,train_samples:]  
           
        # Compute NRMSE between true target values and predictions for the test data
        cpu_nrmse_buffer_blocks[i_b,i_s] = np.sqrt(np.mean((target[:,train_samples:] - pred_buffer)**2,axis=1)/target_var)[0]
        cpu_nrmse_single_1st_2nd_blocks[i_b,i_s] = np.sqrt(np.mean((target[:,train_samples:] - pred_single_1st_2nd)**2,axis=1)/target_var)[0]          
        cpu_nrmse_buffer_2nd_order_blocks[i_b,i_s] = np.sqrt(np.mean((target[:,train_samples:] - pred_buffer_2nd_order)**2,axis=1)/target_var)[0]

# Compute mean NRMSE and its standard errors for all simulations
cpu_nrmse_buffer_blocks_m = np.mean(cpu_nrmse_buffer_blocks, axis=1)
cpu_nrmse_single_1st_2nd_blocks_m = np.mean(cpu_nrmse_single_1st_2nd_blocks, axis=1)
cpu_nrmse_buffer_2nd_order_blocks_m = np.mean(cpu_nrmse_buffer_2nd_order_blocks, axis=1)

# Standard errors
cpu_nrmse_buffer_blocks_low  = cpu_nrmse_buffer_blocks_m - np.std(cpu_nrmse_buffer_blocks, axis=1)/np.sqrt(simul)
cpu_nrmse_buffer_blocks_high  = cpu_nrmse_buffer_blocks_m + np.std(cpu_nrmse_buffer_blocks, axis=1)/np.sqrt(simul)

cpu_nrmse_single_1st_2nd_blocks_low  = cpu_nrmse_single_1st_2nd_blocks_m - np.std(cpu_nrmse_single_1st_2nd_blocks, axis=1)/np.sqrt(simul)
cpu_nrmse_single_1st_2nd_blocks_high  = cpu_nrmse_single_1st_2nd_blocks_m + np.std(cpu_nrmse_single_1st_2nd_blocks, axis=1)/np.sqrt(simul)

cpu_nrmse_buffer_2nd_order_blocks_low  = cpu_nrmse_buffer_2nd_order_blocks_m - np.std(cpu_nrmse_buffer_2nd_order_blocks, axis=1)/np.sqrt(simul)
cpu_nrmse_buffer_2nd_order_blocks_high  = cpu_nrmse_buffer_2nd_order_blocks_m + np.std(cpu_nrmse_buffer_2nd_order_blocks, axis=1)/np.sqrt(simul)
    
## Visualization part ##    
    
##
## Plot for Figure 5b
##
f, axs = plt.subplots(1, 3, figsize=(6,1.4),dpi=500)

# Plot the target
axs[0].plot(target[:,train_samples:-2].T, linewidth=3, color ='gray', label= 'target');
axs[1].plot(target[:,train_samples:-2].T, linewidth=3, color ='gray', label= 'target');
axs[2].plot(target[:,train_samples:-2].T, linewidth=3, color ='gray', label= 'target');

# Plot the predictions
axs[0].plot((pred_buffer_loihi_all_blocks[:,train_samples:]).T, linewidth=1, linestyle = 'solid', color ='g')
axs[1].plot((pred_single_1st_2nd_loihi_all_blocks[:,train_samples:]).T, linewidth=1, linestyle = 'solid', color ='b')
axs[2].plot((pred_buffer_2nd_order_loihi_all_blocks[:,train_samples:]).T, linewidth=1, linestyle = 'solid', color ='c')

axs[0].title.set_text('1st-order \n memory buffer')
axs[1].title.set_text('1st- & 2nd- order \n single time point')
axs[2].title.set_text('1st & 2nd - order \n memory buffer')
axs[0].set_xlabel('Time points')
axs[0].legend(loc='lower left')
axs[1].set_xlabel('Time points')
axs[2].set_xlabel('Time points')
axs[0].set_xticks([0,100,200,300])
axs[1].set_xticks([0,100,200,300])
axs[2].set_xticks([0,100,200,300])
axs[0].text(-57,13.7,'b)', weight='bold', ha='left', va='top')

plt.savefig('/results/Figure_5_b_loihi2_vs_cpu.png',bbox_inches="tight")
plt.show() 

##
## Plot for Figure 5c
##
plt.figure(figsize=(4.,2.),dpi=500)
ax = plt.gca()

# Plot the curves
# Loihi2
plt.plot(np.arange(1,block_num+1), nrmse_buffer_blocks_m ,color ='g', linewidth=1,label= '1st, memory buffer', linestyle = 'solid')
plt.plot(np.arange(1,block_num+1), nrmse_single_1st_2nd_blocks_m ,color ='b', linewidth=1,label= '1st & 2nd, single time point', linestyle = 'solid')
plt.plot(np.arange(1,block_num+1), nrmse_buffer_2nd_order_blocks_m ,color ='c', linewidth=1,label= '1st & 2nd, memory buffer', linestyle = 'solid')
# CPU
plt.plot(np.arange(1,block_num+1), cpu_nrmse_buffer_blocks_m ,color ='g', linewidth=1, linestyle = 'dashed')
plt.plot(np.arange(1,block_num+1), cpu_nrmse_single_1st_2nd_blocks_m ,color ='b', linewidth=1, linestyle = 'dashed')
plt.plot(np.arange(1,block_num+1), cpu_nrmse_buffer_2nd_order_blocks_m ,color ='c', linewidth=1, linestyle = 'dashed')

# Plot the errors
# Loihi2
ax.fill_between(np.arange(1,block_num+1), nrmse_buffer_blocks_low, nrmse_buffer_blocks_high, alpha=0.3,color='g')
ax.fill_between(np.arange(1,block_num+1), nrmse_single_1st_2nd_blocks_low, nrmse_single_1st_2nd_blocks_high, alpha=0.3,color='b')
ax.fill_between(np.arange(1,block_num+1), nrmse_buffer_2nd_order_blocks_low, nrmse_buffer_2nd_order_blocks_high, alpha=0.3,color='c')
# CPU
ax.fill_between(np.arange(1,block_num+1), cpu_nrmse_buffer_blocks_low, cpu_nrmse_buffer_blocks_high, alpha=0.3,color='g')
ax.fill_between(np.arange(1,block_num+1), cpu_nrmse_single_1st_2nd_blocks_low, cpu_nrmse_single_1st_2nd_blocks_high, alpha=0.3,color='b')
ax.fill_between(np.arange(1,block_num+1), cpu_nrmse_buffer_2nd_order_blocks_low, cpu_nrmse_buffer_2nd_order_blocks_high, alpha=0.3,color='c')

plt.xlabel('Number of blocks in representation')
plt.ylabel('NRMSE')
plt.ylim(0.0001,1.0)
plt.yscale('log')
plt.text(-1.35,1.43,'c)', weight='bold', ha='left', va='top')
plt.grid(color='0.95')
plt.legend(loc='center right', fontsize = '8.5')

plt.savefig('/results/Figure_5_c_loihi2_vs_cpu.png',bbox_inches="tight")
plt.show() 