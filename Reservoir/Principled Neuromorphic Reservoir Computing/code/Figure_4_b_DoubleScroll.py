# -*- coding: utf-8 -*-
"""

This script plots the results of experiments reported in panel b of Figure 4.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from scipy import stats

simul = 1000
se_mad_coefficient = 1.2533*1.4826/np.sqrt(simul)

# Load the results of experiments for RC with product representation & distributed representation. As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA = loadmat('/data/Figure_4_b_DoubleScroll_Product_DistributedRepresentation.mat')

Ridgerange = DATA["Ridgerange"][0,:]
Nrange = DATA["Nrange"][0,:]

ind_ridge = 0

# Remove NaNs
STAT_PR_TS = np.nan_to_num(DATA["STAT_PR_TS"], nan=10)
STAT_DR_TS = np.nan_to_num(DATA["STAT_DR_TS"], nan=10)

# Get medians
STAT_PR_TS_m = np.median(STAT_PR_TS[:,0:simul])
STAT_DR_TS_m = np.median(STAT_DR_TS[:,:,0:simul],axis=2)

# Get intervals
STAT_PR_TS_err_low  = STAT_PR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS,axis = 1)
STAT_PR_TS_err_high = STAT_PR_TS_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS,axis = 1)

STAT_DR_TS_err_low  =  STAT_DR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS, axis = 2)
STAT_DR_TS_err_high =  STAT_DR_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS, axis = 2)

# Load the results of experiments for the MLPs
DATA_mlp= loadmat('/data/Figure_4_b_DoubleScroll_MLPRegressor_62.mat')

# Remove NaNs
STAT_MLP_ts = np.nan_to_num(DATA_mlp["STAT_MLP_TS"], nan=10)[:,0,0:simul]

# Get medians
STAT_MLP_ts_m = np.median(STAT_MLP_ts)

# Get intervals
STAT_MLP_ts_err_low  = STAT_MLP_ts_m - se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_ts, axis=1)
STAT_MLP_ts_err_high = STAT_MLP_ts_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_ts, axis=1)

# Load the results of experiments for the ESNs
DATA_esn = loadmat('/data/Figure_4_b_DoubleScroll_Echo_State_Network.mat')

# Remove NaNs
STAT_ESN_ts = np.nan_to_num(DATA_esn["STAT_ESN_TS"], nan=10)

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
plt.plot(Nrange, STAT_PR_TS_m * np.ones((Nrange.size)),color='r', label='Product; ' + r'$D=62$', linestyle='dashed',  linewidth=2)
plt.plot(Nrange, STAT_ESN_ts_m ,color ='k', linewidth=2,label= 'Echo State Network', linestyle='dotted')
ax.plot(Nrange, STAT_MLP_ts_m * np.ones((Nrange.size)),color='b', label='Perceptron; (62, 256, 128, 128)', linestyle='solid',  linewidth=1.5) 

# Plot the errors
ax.fill_between(Nrange, STAT_PR_TS_err_low* np.ones((Nrange.size)), STAT_PR_TS_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')
ax.fill_between(Nrange, STAT_DR_TS_err_low[:,ind_ridge], STAT_DR_TS_err_high[:,ind_ridge], alpha=0.3,color='g')
ax.fill_between(Nrange, STAT_ESN_ts_err_low, STAT_ESN_ts_err_high, alpha=0.3,color='k')
ax.fill_between(Nrange, STAT_MLP_ts_err_low* np.ones((Nrange.size)), STAT_MLP_ts_err_high* np.ones((Nrange.size)), alpha=0.3,color='b')

# Emphasize the interesting configurations
# RC PR configuration
xlimExpl  = (62-.75, 62+.75)
ylimExpl = (STAT_PR_TS_m-0.0013, STAT_PR_TS_m+0.002)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlimExpl[0], ylimExpl[0]), xlimExpl[1] - xlimExpl[0], ylimExpl[1] - ylimExpl[0], linewidth=2, edgecolor='r', facecolor='r',zorder=10)
ax.add_patch(rect)

# MLP configuration
xlimExpl  = (62-.75, 62+.75)
ylimExpl = (STAT_MLP_ts_m-0.015, STAT_MLP_ts_m+0.023)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlimExpl[0], ylimExpl[0]), xlimExpl[1] - xlimExpl[0], ylimExpl[1] - ylimExpl[0], linewidth=2, edgecolor='b', facecolor='b',zorder=10)
ax.add_patch(rect)

# RC DR configuration
# draw the zoomed ellipse
ell1 = mpl.patches.Ellipse((34, STAT_DR_TS_m[np.where(Nrange == 34)[0][0]][0]), 2*0.75, 2*0.0022, linewidth=1.5, edgecolor='g', facecolor='g',zorder=9)
ax.add_patch(ell1)

plt.xlabel('Dimensionality of representations, $D$')
plt.ylabel('NRMSE')
plt.ylim(0.005,3.0)
plt.yscale("log")
plt.text(10.5,4.95,'b)',  weight='bold', ha='left', va='top')
plt.legend(loc='upper right', fontsize="11")
plt.grid(color='0.95')
plt.title('Double-scroll')

plt.savefig('/results/Figure_4_b_DoubleScroll.png',bbox_inches="tight")
plt.show() 