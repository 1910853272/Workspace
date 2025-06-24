# -*- coding: utf-8 -*-
"""

This script plots the results of experiments reported in panel a of Figure 4.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from scipy import stats

simul = 1000
se_mad_coefficient = 1.2533*1.4826/np.sqrt(simul)

# Load the results of experiments for RC with product representation & distributed representation. As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_PR_DR = loadmat('/data/Figure_4_a_Lorenz_Product_DistributedRepresentation.mat')

Ridgerange = DATA_PR_DR["Ridgerange"][0,:]
Nrange = DATA_PR_DR["Nrange"][0,:]

ind_ridge = 0

# Remove NaNs
STAT_PR_TS = np.nan_to_num(DATA_PR_DR["STAT_PR_TS"], nan=10)
STAT_DR_TS = np.nan_to_num(DATA_PR_DR["STAT_DR_TS"], nan=10)

# Get medians
STAT_PR_TS_m = np.median(STAT_PR_TS[:,0:simul])
STAT_DR_TS_m = np.median(STAT_DR_TS[:,:,0:simul],axis=2)

# Get intervals
STAT_PR_TS_err_low  = STAT_PR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS,axis = 1)
STAT_PR_TS_err_high = STAT_PR_TS_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS,axis = 1)

STAT_DR_TS_err_low  =  STAT_DR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS, axis = 2)
STAT_DR_TS_err_high =  STAT_DR_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS, axis = 2)

# Load the results of experiments for the MLPs
DATA_mlp = loadmat('/data/Figure_4_a_Lorenz_MLPRegressor_28.mat')

# Remove NaNs
STAT_MLP_ts = np.nan_to_num(DATA_mlp["STAT_MLP_TS"], nan=10)[:,0,0:simul]

# Get medians
STAT_MLP_ts_m = np.median(STAT_MLP_ts)

# Get intervals
STAT_MLP_ts_err_low  = STAT_MLP_ts_m - se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_ts, axis=1)
STAT_MLP_ts_err_high = STAT_MLP_ts_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_ts, axis=1)

# Load the results of experiments for the ESNs
DATA_esn = loadmat('/data/Figure_4_a_Lorenz_Echo_State_Network.mat')

# Remove NaNs
STAT_ESN_ts = np.nan_to_num(DATA_esn["STAT_ESN_TS"], nan=10)[:,0:simul]

# Get medians
STAT_ESN_ts_m = np.median(STAT_ESN_ts,axis=1)

# Get intervals
STAT_ESN_ts_err_low  = STAT_ESN_ts_m - se_mad_coefficient*stats.median_abs_deviation(STAT_ESN_ts, axis=1)
STAT_ESN_ts_err_high = STAT_ESN_ts_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_ESN_ts, axis=1)

##
## Plot
##
plt.figure(figsize=(5.,3.36),dpi=500)
ax = plt.gca()

# Plot the curves
plt.plot(Nrange, STAT_DR_TS_m[:,ind_ridge] ,color ='g', linewidth=2,label= 'Distributed')
plt.plot(Nrange, STAT_PR_TS_m * np.ones((Nrange.size)),color='r', label='Product; ' + r'$D=28$', linestyle='dashed',  linewidth=2)
plt.plot(Nrange, STAT_ESN_ts_m ,color ='k', linewidth=2,label= 'Echo State Network', linestyle='dotted')
plt.plot(Nrange, STAT_MLP_ts_m * np.ones((Nrange.size)),color='b', label='Perceptron; (28, 64, 64, 32)', linestyle='solid',  linewidth=1.5) # Picked optimal ridge value for NGRC

# Plot the errors
ax.fill_between(Nrange, STAT_PR_TS_err_low* np.ones((Nrange.size)), STAT_PR_TS_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')
ax.fill_between(Nrange, STAT_DR_TS_err_low[:,ind_ridge], STAT_DR_TS_err_high[:,ind_ridge], alpha=0.3,color='g')
ax.fill_between(Nrange, STAT_ESN_ts_err_low, STAT_ESN_ts_err_high, alpha=0.3,color='k')
ax.fill_between(Nrange, STAT_MLP_ts_err_low* np.ones((Nrange.size)), STAT_MLP_ts_err_high* np.ones((Nrange.size)), alpha=0.3,color='b')

# Emphasize the interesting configurations
# RC PR configuration
xlimExpl  = (28-.33, 28+.33)
ylimExpl = (STAT_PR_TS_m-0.0010, STAT_PR_TS_m+0.0015)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlimExpl[0], ylimExpl[0]), xlimExpl[1] - xlimExpl[0], ylimExpl[1] - ylimExpl[0], linewidth=2, edgecolor='r', facecolor='r',zorder=11)
ax.add_patch(rect)

# MLP configuration
xlimExpl  = (28-.33, 28+.33)
ylimExpl = (STAT_MLP_ts_m-0.025, STAT_MLP_ts_m+0.035)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlimExpl[0], ylimExpl[0]), xlimExpl[1] - xlimExpl[0], ylimExpl[1] - ylimExpl[0], linewidth=2, edgecolor='b', facecolor='b',zorder=10)
ax.add_patch(rect)

# RC DR configuration
# draw the zoomed ellipse
ell = mpl.patches.Ellipse((28, STAT_DR_TS_m[np.where(Nrange == 28)[0][0]][0]), 2*0.33, 2*0.0015, linewidth=1, edgecolor='g', facecolor='g',zorder=12)
ax.add_patch(ell)

plt.xlabel('Dimensionality of representations, $D$')
plt.ylabel('NRMSE')
plt.ylim(0.01,8.0)
plt.yscale("log")
plt.text(3.5,13.5,'a)', weight='bold', ha='left', va='top')
plt.grid(color='0.95')
plt.legend(loc='upper right', fontsize="11")
plt.title('Lorenz63')

plt.savefig('/results/Figure_4_a_Lorenz.png',bbox_inches="tight")
plt.show() 