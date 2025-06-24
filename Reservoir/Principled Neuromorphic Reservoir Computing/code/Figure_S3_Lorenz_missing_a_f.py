# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in Figure S.3 (panels a)-f)).

"""

# Initial version of this script comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code


import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import Lorenz63

##
## Data - Lorenz63 system
##
lorenz = Lorenz63(dt=0.05, traintime = 20., testtime=45.,method = 'RK45')   

# calculate standard deviation of z component
zstd = np.std(lorenz.data.y[2,:])

##
## Parameters
##

# input dimension
d=lorenz.data.y.shape[0]        
# number of time delay taps
k = 4
# number of time steps between taps. skip = 1 means take consecutive points
skip = 5
# size of linear part of feature vector (leave out z)
dlin = k*(d-1)

##
## Product representation RC
##

# ridge parameter for regression PR RC
ridge_param = 1e-7

# size of nonlinear part of feature vector
dnonlin = int(dlin*(dlin+1)/2)
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

# create an array to hold the linear part of the feature vector
buffer = np.zeros((dlin,lorenz.maxtime_pts))

# create an array to hold the full feature vector for all time after warmup
# (use ones so the constant term is already 1)
pr_features = np.ones((dtot+1,lorenz.maxtime_pts-lorenz.warmup_pts))

# fill in the linear part of the feature vector for all times
for delay in range(k):
    for j in range(delay,lorenz.maxtime_pts):
        # only include x and y
        buffer[(d-1)*delay:(d-1)*(delay+1),j]=lorenz.data.y[0:2,j-delay*skip]

# copy over the linear part (shift over by one to account for constant)
# unlike forecasting, we can do this all in one shot, and we don't need to
# shift times for one-step-ahead prediction
pr_features[1:dlin+1,:]=buffer[:,lorenz.warmup_pts:lorenz.maxtime_pts]

# fill in the non-linear part
cnt=0
for row in range(dlin):
    for column in range(row,dlin):
        # shift by one for constant
        pr_features[dlin+1+cnt,:]=buffer[row,lorenz.warmup_pts:lorenz.maxtime_pts]*buffer[column,lorenz.warmup_pts:lorenz.maxtime_pts]
        cnt += 1

# ridge regression: train W_out to map out to Lorenz z
W_out = lorenz.data.y[2,lorenz.warmup_pts:lorenz.warmtrain_pts] @ pr_features[:,0:lorenz.traintime_pts].T @ np.linalg.pinv(pr_features[:,0:lorenz.traintime_pts] @ pr_features[:,0:lorenz.traintime_pts].T + ridge_param*np.identity(dtot+1))

# once we have W_out, we can predict the entire shot
# apply W_out to the feature vector to get the output
# this includes both training and testing phases
pr_predict = W_out @ pr_features[:,:]

# calculate NRMSE between true Lorenz z and training output
nrmse_tr_pr = np.sqrt(np.mean((lorenz.data.y[2,lorenz.warmup_pts:lorenz.warmtrain_pts]-pr_predict[0:lorenz.traintime_pts])**2))/zstd    
print('Product representation, train NRMSE ' +str(nrmse_tr_pr))

# calculate NRMSE between true Lorenz z and prediction
nrmse_ts = np.sqrt(np.mean((lorenz.data.y[2,lorenz.warmtrain_pts:lorenz.maxtime_pts]-pr_predict[lorenz.traintime_pts:lorenz.maxtime_pts-lorenz.warmup_pts])**2))/zstd    
print('Product representation, test NRMSE '+str(nrmse_ts))

##
## Distributed representation RC
##
np.random.seed(seed=1)

# ridge parameter for regression DR RC
ridge_param_dr = 1e-7

# Dimensionality of hypervectors
D = 45
D_nb = D-1 # no bias        
perm = np.random.permutation(D_nb)
pos=np.random.randn(dlin, D_nb)
pos=np.fft.fft(pos, axis=1)/np.abs(np.fft.fft(pos, axis=1))
B = np.fft.ifft(pos, axis=1).real

# Normalize according to std of x and y in the trainig part
dr_features=buffer[:,lorenz.warmup_pts:lorenz.maxtime_pts]
std_train = np.std(dr_features[0:2,0:lorenz.traintime_pts], axis=1, keepdims=True)
std_train = np.concatenate([std_train, std_train, std_train, std_train], axis=0) 
dr_features=dr_features/std_train

dr_states_train = np.ones((D,lorenz.traintime_pts))
for i in range(lorenz.traintime_pts):
    OlinHD=np.sum(dr_features[:,i:i+1]*B, axis=0, keepdims=True).real                
    OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,perm])).real
    dr_states_train[1:,i:i+1] = np.transpose(1.*OnlHD+2*OlinHD)    
    
# ridge regression: train W_out to map out to Lorenz z
WoutHD_ridge = lorenz.data.y[2,lorenz.warmup_pts:lorenz.warmtrain_pts] @ dr_states_train[:,:].T @ np.linalg.pinv(dr_states_train[:,:] @ dr_states_train[:,:].T + ridge_param_dr*np.identity(D))

# apply W_out to the training feature vector to get the training output
dr_predict_train = WoutHD_ridge @ dr_states_train[:,0:lorenz.traintime_pts]

# calculate NRMSE between true Lorenz z and training output
nrmse_tr_dr = np.sqrt(np.mean((lorenz.data.y[2,lorenz.warmup_pts:lorenz.warmtrain_pts]-dr_predict_train[0:lorenz.traintime_pts])**2))/zstd  
print('Distributed representation, train NRMSE '+str(nrmse_tr_dr))

dr_states_test = np.ones((D,lorenz.maxtime_pts-lorenz.warmtrain_pts))
for i in range(lorenz.maxtime_pts-lorenz.warmtrain_pts):
    OlinHD=np.sum(dr_features[:,lorenz.traintime_pts+i:lorenz.traintime_pts+i+1]*B, axis=0, keepdims=True).real            
    OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,perm])).real
    dr_states_test[1:,i:i+1] = np.transpose(1.*OnlHD+2*OlinHD)

# apply W_out to the test data vector to get the prediction
dr_predict = WoutHD_ridge @ dr_states_test[:,0:lorenz.maxtime_pts-lorenz.warmtrain_pts]

# calculate NRMSE between true Lorenz z and prediction
nrmse_ts_dr = np.sqrt(np.mean((lorenz.data.y[2,lorenz.warmtrain_pts:lorenz.maxtime_pts]-dr_predict[0:lorenz.maxtime_pts-lorenz.warmtrain_pts])**2))/zstd   
print('Distributed representation, test NRMSE '+str(nrmse_ts_dr))


##
## Plot
##
t_linewidth=.8
fig1 = plt.figure(dpi=300)
fig1.set_figheight(8)
fig1.set_figwidth(12)
h=240
w=2

# top left of grid is 0,0
axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 0), colspan=1, rowspan=30) 
axs2 = plt.subplot2grid(shape=(h,w), loc=(36, 0), colspan=1, rowspan=30)
axs3 = plt.subplot2grid(shape=(h,w), loc=(72, 0), colspan=1, rowspan=30)
axs4 = plt.subplot2grid(shape=(h,w), loc=(132, 0), colspan=1, rowspan=30)
axs5 = plt.subplot2grid(shape=(h,w), loc=(168, 0), colspan=1, rowspan=30)
axs6 = plt.subplot2grid(shape=(h,w), loc=(204, 0),colspan=1, rowspan=30)

# training phase x
axs1.set_title('training phase')
axs1.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,lorenz.data.y[0,lorenz.warmup_pts:lorenz.warmtrain_pts],color='k',linewidth=t_linewidth)
axs1.set_ylabel('x', style='italic')
axs1.axes.xaxis.set_ticklabels([])
axs1.axes.set_xbound(-.08,lorenz.traintime+.05)
axs1.axes.set_ybound(-21.,21.)
axs1.text(-.14,.9,'a)', weight='bold', ha='left', va='bottom',transform=axs1.transAxes)

# training phase y
axs2.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,lorenz.data.y[1,lorenz.warmup_pts:lorenz.warmtrain_pts],color='k',linewidth=t_linewidth)
axs2.set_ylabel('y', style='italic')
axs2.axes.xaxis.set_ticklabels([])
axs2.axes.set_xbound(-.08,lorenz.traintime+.05)
axs2.axes.set_ybound(-26.,26.)
axs2.text(-.14,.9,'b)', weight='bold', ha='left', va='bottom',transform=axs2.transAxes)

# training phase z
axs3.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,lorenz.data.y[2,lorenz.warmup_pts:lorenz.warmtrain_pts],color='k',linewidth=3*t_linewidth, linestyle='dotted')
axs3.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,pr_predict[0:lorenz.traintime_pts],color='r',linewidth=t_linewidth, linestyle='dashed')
axs3.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,dr_predict_train[0:lorenz.traintime_pts],color='g',linewidth=t_linewidth, linestyle='dashdot')
axs3.set_ylabel('z', style='italic')
axs3.set_xlabel('Time')
axs3.axes.set_xbound(-.08,lorenz.traintime+.05)
axs3.axes.set_ybound(3.,48.)
axs3.text(-.14,.9,'c)', weight='bold', ha='left', va='bottom',transform=axs3.transAxes)

# testing phase x
axs4.set_title('forecasting phase')
axs4.plot(lorenz.t_eval[lorenz.warmtrain_pts:lorenz.maxtime_pts]-lorenz.warmup,lorenz.data.y[0,lorenz.warmtrain_pts:lorenz.maxtime_pts],color='k',linewidth=t_linewidth)
axs4.set_ylabel('x', style='italic')
axs4.axes.xaxis.set_ticklabels([])
axs4.axes.set_ybound(-21.,21.)
axs4.axes.set_xbound(lorenz.traintime-.5,lorenz.maxtime-lorenz.warmup+.5)
axs4.text(-.14,.9,'d)', weight='bold', ha='left', va='bottom',transform=axs4.transAxes)

# testing phase y
axs5.plot(lorenz.t_eval[lorenz.warmtrain_pts:lorenz.maxtime_pts]-lorenz.warmup,lorenz.data.y[1,lorenz.warmtrain_pts:lorenz.maxtime_pts],color='k',linewidth=t_linewidth)
axs5.set_ylabel('y', style='italic')
axs5.axes.xaxis.set_ticklabels([])
axs5.axes.set_ybound(-26.,26.)
axs5.axes.set_xbound(lorenz.traintime-.5,lorenz.maxtime-lorenz.warmup+.5)
axs5.text(-.14,.9,'e)', weight='bold', ha='left', va='bottom',transform=axs5.transAxes)

# testing phose z
axs6.plot(lorenz.t_eval[lorenz.warmtrain_pts:lorenz.maxtime_pts]-lorenz.warmup,lorenz.data.y[2,lorenz.warmtrain_pts:lorenz.maxtime_pts],color='k',linewidth=3*t_linewidth, linestyle='dotted')
axs6.plot(lorenz.t_eval[lorenz.warmtrain_pts:lorenz.maxtime_pts]-lorenz.warmup,pr_predict[lorenz.traintime_pts:lorenz.maxtime_pts-lorenz.warmup_pts],color='r',linewidth=t_linewidth, linestyle='dashed')
axs6.plot(lorenz.t_eval[lorenz.warmtrain_pts:lorenz.maxtime_pts]-lorenz.warmup,dr_predict[0:lorenz.maxtime_pts-lorenz.warmtrain_pts],color='g',linewidth=t_linewidth, linestyle='dashdot')
axs6.set_ylabel('z', style='italic')
axs6.set_xlabel('Time')
axs6.axes.set_ybound(3.,48.)
axs6.axes.set_xbound(lorenz.traintime-.5,lorenz.maxtime-lorenz.warmup+.5)
axs6.text(-.14,.9,'f)', weight='bold', ha='left', va='bottom',transform=axs6.transAxes)


fig1.align_ylabels([axs1,axs2,axs3,axs4,axs5,axs6])
plt.savefig('/results/Figure_S3_Lorenz_missing_a_f.png')
plt.show() 
