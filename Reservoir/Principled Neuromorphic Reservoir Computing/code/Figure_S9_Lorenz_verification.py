# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in Figure S.9.

"""

# Initial version of this script comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.signal
import scipy.interpolate
from scipy.optimize import fsolve
from DataGeneration import Lorenz63
from ProductRepresentation import ProductRepresentationRC
from DistributedRepresentation import DistributedRepresentationRC

# use interpolating splines to find maxima of input signal, and return an array of (M_i, M_i+1) pairs
def return_map_spline(v):
    spline = scipy.interpolate.InterpolatedUnivariateSpline(np.arange(len(v)), v, k=4)
    spline_d = spline.derivative()
    spline_dd = spline_d.derivative()

    # when is the derivative of v zero?
    extimes = spline_d.roots()

    # discard times out of bound
    extimes = extimes[extimes > 0]
    extimes = extimes[extimes < len(v) - 1]

    # select only local maxima
    extimes = extimes[spline_dd(extimes) < 0]

    # find values
    ex = spline(extimes)

    # construct return map
    return np.stack([ex[:-1], ex[1:]], axis=-1)

# These functions do a single step prediction for a trial fixed point and return the difference between the input and prediction
# We can then solve func(p_fp) == 0 to find a fixed point p_fp
def func_pr(p_fp):
    # create a trial input feature vector
    out_vec=np.ones(productRepresentation.dtot)
    # fill in the linear part
    for ii in range(productRepresentation.k):
        # all past input is p_fp
        out_vec[int(productRepresentation.bias)+ii*productRepresentation.d:1+(ii+1)*productRepresentation.d]=p_fp[0:productRepresentation.d]
    # fill in the nonlinear part of the feature vector
    cnt=0
    for row in range(productRepresentation.dlin):
        for column in range(row,productRepresentation.dlin):
            out_vec[int(productRepresentation.bias)+productRepresentation.dlin+cnt]=out_vec[1+row]*out_vec[1+column]
            cnt += 1
    return productRepresentation.W_out @ out_vec

def func_dr(p_fp):
    # linear features to form initial encoding into a high-dimensional space
    inp=np.expand_dims(np.concatenate([p_fp,p_fp], axis=0), axis=1)
    inp=inp/distributedRepresentation.std_train

    # create a randomized representation       
    OlinHD=np.sum(inp*distributedRepresentation.B, axis=0, keepdims=True).real
    OnlHD = np.fft.ifft (np.fft.fft(OlinHD)*np.fft.fft(OlinHD[:,distributedRepresentation.perm])).real    
    
    Otot = np.ones((distributedRepresentation.D,1))
    Otot[int(distributedRepresentation.bias):,0:1] = np.transpose(1.*OnlHD+1.*OlinHD)                  
    return distributedRepresentation.W_out @ Otot[:,0]

##
## Data - Lorenz63 system
##
lorenz = Lorenz63()   

# setup variables for predicted and true fixed points
# true fixed point 0 is 0
t_fp0=np.zeros(3)
t_fp1=np.zeros(3)
t_fp2=np.zeros(3)
# true fixed point 1 is at...
t_fp1[0]=np.sqrt(lorenz.betaLor*(lorenz.rho-1))
t_fp1[1]=np.sqrt(lorenz.betaLor*(lorenz.rho-1))
t_fp1[2]=lorenz.rho-1
# true fixed point 2 is at...
t_fp2[0]=-t_fp1[0]
t_fp2[1]=-t_fp1[1]
t_fp2[2]=t_fp1[2]

##
## Parameters
##

# number of NRMSE trials
simul=100
# how far in to Lorenz solution to start
start=5.
traintime = 10.
# ridge parameter for regression

# create a vector of warmup times to use, dividing space into
# simul segments of length traintime
warmup_v=np.arange(start,traintime*simul+start,traintime)

# storage for results
n_fp1_diff_v=np.zeros(simul)
n_fp2_diff_v=np.zeros(simul)
n_fp0_diff_v=np.zeros(simul)
p_fp1_norm_v=np.zeros((simul, 3))
p_fp2_norm_v=np.zeros((simul, 3))
p_fp0_norm_v=np.zeros((simul, 3))


n_fp1_diff_hd_v=np.zeros(simul)
n_fp2_diff_hd_v=np.zeros(simul)
n_fp0_diff_hd_v=np.zeros(simul)
p_fp1_norm_hd_v=np.zeros((simul, 3))
p_fp2_norm_hd_v=np.zeros((simul, 3))
p_fp0_norm_hd_v=np.zeros((simul, 3))


# run a trial with the given warmup time
# run many trials and collect the results
for i in range(simul):
    
    warmup = warmup_v[i]
    
    ##
    ## Data - Lorenz63 system
    ##
    lorenz = Lorenz63(warmup= warmup_v[i], traintime = traintime, testtime=0.)    

    ##
    ## RC with product representation
    ##
    # ridge parameter for regression
    ridge_param = 2.5e-6
    
    productRepresentation = ProductRepresentationRC(system=lorenz, k=2, ridge_param=ridge_param)
    productRepresentation.fit12()

    ##
    ## RC with distributed representation 
    ##        
    # ridge parameter for regression
    ridge_param_dr = 1.e-6
    
    distributedRepresentation = DistributedRepresentationRC(system=lorenz, D=28, k=2, ridge_param=ridge_param_dr, seed = i)
    distributedRepresentation.fit12()
  
    
    # solve for the first fixed point and calculate distances
    p_fp1 = fsolve(func_pr, t_fp1)
    n_fp1_diff_v[i]=np.sqrt(np.sum((t_fp1-p_fp1)**2)/lorenz.total_var)
    p_fp1_norm_v[i] = (t_fp1 - p_fp1) / np.sqrt(lorenz.total_var)

    p_fp1_hd = fsolve(func_dr, t_fp1)
    n_fp1_diff_hd_v[i]=np.sqrt(np.sum((t_fp1-p_fp1_hd)**2)/lorenz.total_var)
    p_fp1_norm_hd_v[i] = (t_fp1 - p_fp1_hd) / np.sqrt(lorenz.total_var)

    # solve for second fixed point
    p_fp2 = fsolve(func_pr, t_fp2)
    n_fp2_diff_v[i]=np.sqrt(np.sum((t_fp2-p_fp2)**2)/lorenz.total_var)
    p_fp2_norm_v[i] = (t_fp2 - p_fp2) / np.sqrt(lorenz.total_var)

    p_fp2_hd = fsolve(func_dr, t_fp2)
    n_fp2_diff_hd_v[i]=np.sqrt(np.sum((t_fp2-p_fp2_hd)**2)/lorenz.total_var)
    p_fp2_norm_hd_v[i] = (t_fp2 - p_fp2_hd) / np.sqrt(lorenz.total_var)

    # solve for 0 fixed point
    p_fp0=fsolve(func_pr, t_fp0)
    n_fp0_diff_v[i]=np.sqrt(np.sum((t_fp0-p_fp0)**2)/lorenz.total_var)
    p_fp0_norm_v[i] = (t_fp0 - p_fp0) / np.sqrt(lorenz.total_var)

    p_fp0_hd = fsolve(func_dr, t_fp0)
    n_fp0_diff_hd_v[i]=np.sqrt(np.sum((t_fp0-p_fp0_hd)**2)/lorenz.total_var)
    p_fp0_norm_hd_v[i] = (t_fp0-p_fp0_hd) / np.sqrt(lorenz.total_var)

# mean / err of (normalized L2 distance from true to predicted fixed point)
## RC with product representation
print()
print('RC-PR: mean, meanerr, fp1 nL2 distance: '+str(np.mean(n_fp1_diff_v))+' '+str(np.std(n_fp1_diff_v)/np.sqrt(simul)))
print('RC-PR: mean, meanerr, fp2 nL2 distance: '+str(np.mean(n_fp2_diff_v))+' '+str(np.std(n_fp2_diff_v)/np.sqrt(simul)))
print('RC-PR: mean, meanerr, fp0 nL2 distance: '+str(np.mean(n_fp0_diff_v))+' '+str(np.std(n_fp0_diff_v)/np.sqrt(simul)))
## RC with distributed representation
print()
print('RC-DR: mean, meanerr, fp1 nL2 distance: '+str(np.mean(n_fp1_diff_hd_v))+' '+str(np.std(n_fp1_diff_hd_v)/np.sqrt(simul)))
print('RC-DR: mean, meanerr, fp2 nL2 distance: '+str(np.mean(n_fp2_diff_hd_v))+' '+str(np.std(n_fp2_diff_hd_v)/np.sqrt(simul)))
print('RC-DR: mean, meanerr, fp0 nL2 distance: '+str(np.mean(n_fp0_diff_hd_v))+' '+str(np.std(n_fp0_diff_hd_v)/np.sqrt(simul)))

# mean / err of (normalized difference between true and predicted fixed point)
## RC with product representation
print()
print('RC-PR: mean, meanerr, fp1', np.mean(p_fp1_norm_v, axis=0), np.std(p_fp1_norm_v, axis=0) / np.sqrt(simul))
print('RC-PR: mean, meanerr, fp2', np.mean(p_fp2_norm_v, axis=0), np.std(p_fp2_norm_v, axis=0) / np.sqrt(simul))
print('RC-PR: mean, meanerr, fp0', np.mean(p_fp0_norm_v, axis=0), np.std(p_fp0_norm_v, axis=0) / np.sqrt(simul))
## RC with distributed representation
print()
print('RC-DR: mean, meanerr, fp1', np.mean(p_fp1_norm_hd_v, axis=0), np.std(p_fp1_norm_hd_v, axis=0) / np.sqrt(simul))
print('RC-DR: mean, meanerr, fp2', np.mean(p_fp2_norm_hd_v, axis=0), np.std(p_fp2_norm_hd_v, axis=0) / np.sqrt(simul))
print('RC-DR: mean, meanerr, fp0', np.mean(p_fp0_norm_hd_v, axis=0), np.std(p_fp2_norm_hd_v, axis=0) / np.sqrt(simul))

# normalized L2 distance between true and (mean of predicted fixed point)
## RC with product representation
print()
print('RC-PR: nL2 distance to mean, meanerr, fp1', np.sqrt(np.sum(np.mean(p_fp1_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp1_norm_v, axis=0)) / simul))
print('RC-PR: nL2 distance to mean, meanerr, fp2', np.sqrt(np.sum(np.mean(p_fp2_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp2_norm_v, axis=0)) / simul))
print('RC-PR: nL2 distance to mean, meanerr, fp0', np.sqrt(np.sum(np.mean(p_fp0_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp0_norm_v, axis=0)) / simul))
## RC with distributed representation
print()
print('RC-DR: nL2 distance to mean, meanerr, fp1', np.sqrt(np.sum(np.mean(p_fp1_norm_hd_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp1_norm_hd_v, axis=0)) / simul))
print('RC-DR: nL2 distance to mean, meanerr, fp2', np.sqrt(np.sum(np.mean(p_fp2_norm_hd_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp2_norm_hd_v, axis=0)) / simul))
print('RC-DR: nL2 distance to mean, meanerr, fp0', np.sqrt(np.sum(np.mean(p_fp0_norm_hd_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp0_norm_hd_v, axis=0)) / simul))

##
## Data - Lorenz63 system
##
lorenz = Lorenz63(testtime=1000.)

# calculate mean, min, and max for all three components of Lorenz solution
lorenz_stats=np.zeros((3,3))
for i in range(3):
    lorenz_stats[0,i]=np.mean(lorenz.data.y[i,lorenz.warmtrain_pts:lorenz.maxtime_pts])
    lorenz_stats[1,i]=np.min(lorenz.data.y[i,lorenz.warmtrain_pts:lorenz.maxtime_pts])
    lorenz_stats[2,i]=np.max(lorenz.data.y[i,lorenz.warmtrain_pts:lorenz.maxtime_pts])


# run a trial with the given warmup time, and make a map plot

##
## RC with product representation
##
# ridge parameter for regression
ridge_param = 2.5e-6

productRepresentation = ProductRepresentationRC(system=lorenz, k=2, ridge_param=ridge_param)
productRepresentation.fit12()
productRepresentation.predict12()
print('Product representation, test NRMSE: '+str(productRepresentation.nrmse_test))

##
## RC with distributed representation 
##        
# ridge parameter for regression
ridge_param_dr = 1.e-6

distributedRepresentation = DistributedRepresentationRC(system=lorenz, D=28, k=2, ridge_param=ridge_param_dr, seed = 8)
distributedRepresentation.fit12()
distributedRepresentation.predict12()

print('Distributed representation, test NRMSE: '+str(distributedRepresentation.nrmse_test))


##
## Plot
##
# get true return map
rm_cmp = return_map_spline(lorenz.data.y[2,:lorenz.testtime_pts])
# get predicted return map for Product Representration
PR_rm = return_map_spline(productRepresentation.prediction[2, :])
# get predicted return map for Distributed Representration
DR_rm = return_map_spline(distributedRepresentation.prediction[2, :])        

# plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=600, figsize=(7, 6))

# Product Representration
# whole return map
ax1.scatter(rm_cmp[:, 0], rm_cmp[:, 1], marker='P', s=2, color='black', linewidths=0)
ax1.scatter(PR_rm[:, 0], PR_rm[:, 1], marker='X', s=2,  color='red', linewidths=0)
ax1.set_xlim(30, 48)
ax1.set_ylim(30, 48)
ax1.set_xlabel('$M_i$')
ax1.set_ylabel('$M_{i+1}$')
ax1.set_title('Product; ' + r'$D=28$')

# zoomed return map
ax3.scatter(rm_cmp[:, 0], rm_cmp[:, 1], marker='P', s=5,  color='black', linewidths=0)
ax3.scatter(PR_rm[:, 0], PR_rm[:, 1], marker='X', s=5, color='red', linewidths=0)
xlim2 = (34.6, 35.5)
ylim2 = (35.7, 36.6)
ax3.set_xlim(*xlim2)
ax3.set_ylim(*ylim2)
ax3.set_xlabel('$M_i$')
ax3.set_ylabel('$M_{i+1}$')

# Distributed Representration
# whole return map
ax2.scatter(rm_cmp[:, 0], rm_cmp[:, 1], marker='P', s=2, color='black', linewidths=0)
ax2.scatter(DR_rm[:, 0], DR_rm[:, 1], marker='o', s=2,  color='green', linewidths=0)
ax2.set_xlim(30, 48)
ax2.set_ylim(30, 48)
ax2.set_xlabel('$M_i$')
ax2.set_ylabel('$M_{i+1}$')
#ax2.set_title('Implicit realization,\nrandomized; ' + r'$N=28$')
ax2.set_title('Distributed; ' + r'$D=28$')

# zoomed return map
ax4.scatter(rm_cmp[:, 0], rm_cmp[:, 1], marker='P', s=5, color='black', linewidths=0)
ax4.scatter(DR_rm[:, 0], DR_rm[:, 1], marker='o', s=5,  color='green', linewidths=0)
ax4.set_xlim(*xlim2)
ax4.set_ylim(*ylim2)
ax4.set_xlabel('$M_i$')
ax4.set_ylabel('$M_{i+1}$')

# draw the zoomed rectangle on the whole
rect = matplotlib.patches.Rectangle((xlim2[0], ylim2[0]), xlim2[1] - xlim2[0], ylim2[1] - ylim2[0], linewidth=1, edgecolor='k', facecolor='none')
rect2 = matplotlib.patches.Rectangle((xlim2[0], ylim2[0]), xlim2[1] - xlim2[0], ylim2[1] - ylim2[0], linewidth=1, edgecolor='k', facecolor='none')
ax1.add_patch(rect)
ax2.add_patch(rect2)

# subplot labels
ax1.text(-0.25, 1.05, 'a)', weight='bold', transform=ax1.transAxes, fontsize=10, va='top', ha='right')
ax2.text(-0.25, 1.05, 'b)',weight='bold', transform=ax2.transAxes, fontsize=10, va='top', ha='right')
ax3.text(-0.25, 1.05, 'c)',weight='bold', transform=ax3.transAxes, fontsize=10, va='top', ha='right')
ax4.text(-0.25, 1.05, 'd)',weight='bold', transform=ax4.transAxes, fontsize=10, va='top', ha='right')
fig.align_ylabels([ax1,ax3])
fig.align_ylabels([ax2,ax4])

plt.tight_layout()
plt.savefig("/results/Figure_S9_Lorenz_rmap.png") 
plt.show()
