# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in Figure S.3 (panel g)).

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from scipy import stats

simul = 1000
se_mad_coefficient = 1.2533*1.4826/np.sqrt(simul)

# Load the results of experiments for RC with distributed representation.  As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_DR = loadmat('/data/Figure_S3_Lorenz_Z_DR.mat')
Nrange = DATA_DR["Nrange"][0,:]
STAT_DR_TS = DATA_DR["STAT_DR_TS"]

# Get medians
STAT_DR_TS_m = np.median(STAT_DR_TS[:,0:simul],axis=1)
# Get intervals
STAT_DR_TS_err_low  =  STAT_DR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS, axis = 1)
STAT_DR_TS_err_high =  STAT_DR_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS, axis = 1)

# Load the results of experiments for RC with product representation.  As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_PR = loadmat('/data/Figure_S3_Lorenz_Z_PR.mat')
STAT_PR_TS = DATA_PR["STAT_PR_TS"]

# Get medians
STAT_PR_TS_m = np.median(STAT_PR_TS[:,0:simul])
# Get intervals
STAT_PR_TS_err_low  = STAT_PR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS,axis = 1)
STAT_PR_TS_err_high = STAT_PR_TS_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS,axis = 1)

# Load the results of experiments for Echo State Networks
DATA_ESN = loadmat('/data/Figure_S3_Lorenz_Z_Echo_State_Network.mat')

# Remove NaNs
STAT_ESN_TS = np.nan_to_num(DATA_ESN["STAT_ESN_TS"], nan=1000)[:,0:simul]

# Get medians
STAT_ESN_TS_m = np.median(STAT_ESN_TS,axis=1)
# Get intervals
STAT_ESN_TS_err_low  = STAT_ESN_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_ESN_TS, axis=1)
STAT_ESN_TS_err_high = STAT_ESN_TS_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_ESN_TS, axis=1)

##
## Plot
##
plt.figure(figsize=(5.,3.36),dpi=500)
ax = plt.gca()

# Distributed representation
plt.plot(Nrange, STAT_DR_TS_m ,color ='g', linewidth=2,label= 'Distributed')
# Plot the errors
ax.fill_between(Nrange, STAT_DR_TS_err_low, STAT_DR_TS_err_high, alpha=0.3,color='g')

# Product representation
plt.plot(Nrange, STAT_PR_TS_m * np.ones((Nrange.size)),color='r', label='Product; ' + r'$D=45$', linestyle='dashed',  linewidth=2)

# Plot the errors
ax.fill_between(Nrange, STAT_PR_TS_err_low* np.ones((Nrange.size)), STAT_PR_TS_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')

# Echo State Network
plt.plot(Nrange, STAT_ESN_TS_m ,color ='k', linewidth=2,label= 'Echo State Network', linestyle='dashdot')

# Plot the errors
ax.fill_between(Nrange, STAT_ESN_TS_err_low, STAT_ESN_TS_err_high, alpha=0.3,color='k')

# Indicate configurations
xlimExpl  = (45-0.7, 45+0.7)
ylimExpl = (STAT_PR_TS_m-0.007, STAT_PR_TS_m+0.007)
# draw the zoomed rectangle on the whole
rect2 = mpl.patches.Rectangle((xlimExpl[0], ylimExpl[0]), xlimExpl[1] - xlimExpl[0], ylimExpl[1] - ylimExpl[0], linewidth=2, edgecolor='r', facecolor='r',zorder=10)
ax.add_patch(rect2)

ell1 = mpl.patches.Ellipse((45, STAT_DR_TS_m[np.where(Nrange == 45)[0][0]]), 2*0.7, 2*0.008, linewidth=1, edgecolor='g', facecolor='g',zorder=10)
ax.add_patch(ell1)

plt.xlabel('Dimensionality of representations, $D$')
plt.ylabel('NRMSE')
plt.ylim(0.0,0.5)
plt.xlim(20.0,100.0)

plt.text(7.,.52,'g)', weight='bold', ha='left', va='top')
plt.grid(color='0.95')
plt.legend(loc='upper right')

plt.savefig('/results/Figure_S3_Lorenz_missing_g.png',bbox_inches="tight")
plt.show() 