# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in Figure S.10.

"""

# Initial version of this script comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code

import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import Lorenz63
from ProductRepresentation import ProductRepresentationRC
from DistributedRepresentation import DistributedRepresentationRC
import warnings
#suppress warnings to account for the situations when the solutions diverge
warnings.filterwarnings('ignore')

##
## Parameters
##

# ridge parameter for regression
ridge_param = 2.5e-6

# ridge parameter for regression
ridge_param_dr = 1.e-7

# Number of simulations to run
simul = 100
seeds = list(range(0, simul))  # skip some seeds resulting in oveflow for NG-RC

npts_train=21
# How far into Lorenz solution to start
start_train=4.
end_train=24.

step_train=(end_train-start_train)/(npts_train-1)

# Create a vector of train times to use, dividing space into simul segments of length traintime
traintime_v=np.arange(start_train,end_train+step_train,step_train)

# Medians
STAT_PR_TS=np.empty(npts_train)
STAT_DR_TS=np.empty(npts_train)

# Intervals
STAT_PR_TS_err_low=np.empty(npts_train)
STAT_PR_TS_err_high=np.empty(npts_train)
STAT_DR_TS_err_low=np.empty(npts_train)
STAT_DR_TS_err_high=np.empty(npts_train)

# Run many trials and collect the results
for j in range(npts_train):
    print('Training step', j)
    # Intermediate storage of the results
    STAT_PR_TS_v=np.zeros(simul)
    STAT_DR_TS_v=np.zeros(simul)
    for i in range(simul):
        ##
        ## Data - Lorenz63 system
        ##
        lorenz = Lorenz63(warmup = 10., traintime = traintime_v[j], seed = seeds[i])

        ##
        ## RC with product representation
        ##
        productRepresentation = ProductRepresentationRC(system=lorenz, k=2, ridge_param=ridge_param)
        productRepresentation.fit12()
        productRepresentation.predict12()
        STAT_PR_TS_v[i] = productRepresentation.nrmse_test

        ##
        ## RC with distributed representation 
        ##        
        distributedRepresentation = DistributedRepresentationRC(system=lorenz, D=28, k=2, ridge_param=ridge_param_dr, seed = seeds[i])
        distributedRepresentation.fit12()
        distributedRepresentation.predict12()
        STAT_DR_TS_v[i] = distributedRepresentation.nrmse_test

    # Remove NaNs
    STAT_PR_TS_v = np.nan_to_num(STAT_PR_TS_v, nan=10)
    STAT_DR_TS_v = np.nan_to_num(STAT_DR_TS_v, nan=10)
    
    # Get medians
    STAT_PR_TS[j]=np.median(STAT_PR_TS_v)
    STAT_PR_TS_err_low[j]  = STAT_PR_TS[j] - np.percentile(STAT_PR_TS_v, 25) 
    STAT_PR_TS_err_high[j] = np.percentile(STAT_PR_TS_v, 75) - STAT_PR_TS[j]
 
    # Get intervals
    STAT_DR_TS[j]=np.median(STAT_DR_TS_v)
    STAT_DR_TS_err_low[j]  = STAT_DR_TS[j] - np.percentile(STAT_DR_TS_v, 25)
    STAT_DR_TS_err_high[j] = np.percentile(STAT_DR_TS_v, 75) - STAT_DR_TS[j]
    
##
## Plot
##
plt.figure(figsize=(6.,2.33),dpi=300)
plt.errorbar(traintime_v/lorenz.dt,STAT_DR_TS[:],yerr=np.vstack([STAT_DR_TS_err_low, STAT_DR_TS_err_high]),color='g', linestyle='dashed',label='Distributed; ' +r'$D=28$',linewidth=3)
plt.errorbar(traintime_v/lorenz.dt,STAT_PR_TS[:],yerr=np.vstack([STAT_PR_TS_err_low, STAT_PR_TS_err_high]) ,color='r', label='Product; ' +r'$D=28$', linewidth=2)


plt.xlabel('Number of training points, ' +r'$r$')
plt.xlim(100.,1000.)
plt.xticks([200,400,600,800,1000])
plt.ylabel('NRMSE')
plt.ylim(0.,0.2)
plt.grid(color='0.95')
plt.legend()

plt.savefig('/results/Figure_S10_Lorenz_NRMSE_vs_training.png',bbox_inches="tight")
plt.show()