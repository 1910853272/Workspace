# -*- coding: utf-8 -*-
"""

This script plots the results of experiments reported in panel a of Figure 4.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from scipy import stats

simul = 100
se_mad_coefficient = 1.2533*1.4826/np.sqrt(simul)

# Load the results of experiments for the ESNs with linear activation. As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_esn_lin = loadmat('/data/Figure_S2_Kuramoto_Sivashinsky_EchoStateNetwork_lin.mat')

# Remove NaNs
STAT_ESN_lin_ts = np.nan_to_num(DATA_esn_lin["STAT_ESN_TS"], nan=1000)[:,0:simul]

# Get medians
STAT_ESN_lin_ts_m = np.median(STAT_ESN_lin_ts,axis=1)

# Get intervals
STAT_ESN_lin_ts_err_low  = STAT_ESN_lin_ts_m - se_mad_coefficient*stats.median_abs_deviation(STAT_ESN_lin_ts, axis=1)
STAT_ESN_lin_ts_err_high = STAT_ESN_lin_ts_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_ESN_lin_ts, axis=1)


# Load the results of experiments for the ESNs with tanh activation. As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_esn = loadmat('/data/Figure_S2_Kuramoto_Sivashinsky_EchoStateNetwork_tanh.mat')

# Remove NaNs
STAT_ESN_ts = np.nan_to_num(DATA_esn["STAT_ESN_TS"], nan=1000)[:,0:simul]

# Get medians
STAT_ESN_ts_m = np.median(STAT_ESN_ts,axis=1)

# Get intervals
STAT_ESN_ts_err_low  = STAT_ESN_ts_m - se_mad_coefficient*stats.median_abs_deviation(STAT_ESN_ts, axis=1)
STAT_ESN_ts_err_high = STAT_ESN_ts_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_ESN_ts, axis=1)


# Load the results of experiments for RC with product representation.  As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_PR_1 = loadmat('/data/Figure_S2_Kuramoto_Sivashinsky_ProductRepresentation_1.mat')
# Remove NaNs
STAT_PR_TS_1 = np.nan_to_num(DATA_PR_1["STAT_PR_TS"], nan=1000)
# Get medians
STAT_PR_TS_1_m = np.median(STAT_PR_TS_1[:,0:simul])

# Get intervals
STAT_PR_TS_1_err_low  = STAT_PR_TS_1_m - se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS_1,axis = 1)
STAT_PR_TS_1_err_high = STAT_PR_TS_1_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS_1,axis = 1)

# Load the results of experiments for RC with product representation.  As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_PR = loadmat('/data/Figure_S2_Kuramoto_Sivashinsky_ProductRepresentation_123.mat')
# Remove NaNs
STAT_PR_TS = np.nan_to_num(DATA_PR["STAT_PR_TS"], nan=1000)
# Get medians
STAT_PR_TS_m = np.median(STAT_PR_TS[:,0:simul])

# Get intervals
STAT_PR_TS_err_low  = STAT_PR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS,axis = 1)
STAT_PR_TS_err_high = STAT_PR_TS_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS,axis = 1)


# Load the results of experiments for RC with distributed representation.  As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_DR = loadmat('/data/Figure_S2_Kuramoto_Sivashinsky_DistributedRepresentation_123.mat')
Nrange = DATA_DR["Drange"][0,:]
# Remove NaNs
STAT_DR_TS = np.nan_to_num(DATA_DR["STAT_DR_TS"], nan=1000)
# Get medians
STAT_DR_TS_m = np.median(STAT_DR_TS[:,0:simul],axis=1)

# Get intervals
STAT_DR_TS_err_low  =  STAT_DR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS, axis = 1)
STAT_DR_TS_err_high =  STAT_DR_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS, axis = 1)


##
## Plot
##
plt.rcParams.update({'font.size': 13})
plt.figure(figsize=(7.,3.36),dpi=500)
ax = plt.gca()

# Plot the curves
plt.plot(Nrange, STAT_ESN_lin_ts_m ,color ='k', linewidth=2,label= 'Echo State Network; ' + r'$g(x) = x$', linestyle='dashed')
plt.plot(Nrange, STAT_ESN_ts_m ,color ='k', linewidth=2,label= 'Echo State Network; ' + r'$g(x) = \tanh(x)$', linestyle='solid')
plt.plot(Nrange, STAT_PR_TS_1_m * np.ones((Nrange.size)),color='r', label='Product; ' + r'$\mathcal{T} = (0)$', linestyle='dashdot',  linewidth=2)
plt.plot(Nrange, STAT_PR_TS_m * np.ones((Nrange.size)),color='r', label='Product; ' + r'$\mathcal{T} = (0,1,2)$', linestyle='solid',  linewidth=2)
plt.plot(Nrange, STAT_DR_TS_m ,color ='g', linewidth=2,label= 'Distributed; ' + r'$\mathcal{T} = (0,1,2)$')

# Plot the errors
ax.fill_between(Nrange, STAT_PR_TS_1_err_low* np.ones((Nrange.size)), STAT_PR_TS_1_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')
ax.fill_between(Nrange, STAT_PR_TS_err_low* np.ones((Nrange.size)), STAT_PR_TS_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')
ax.fill_between(Nrange, STAT_DR_TS_err_low, STAT_DR_TS_err_high, alpha=0.3,color='g')
ax.fill_between(Nrange, STAT_ESN_ts_err_low, STAT_ESN_ts_err_high, alpha=0.3,color='k')
ax.fill_between(Nrange, STAT_ESN_lin_ts_err_low, STAT_ESN_lin_ts_err_high, alpha=0.3,color='k')

# Emphasize the interesting configurations
xlimExpl  = (4000-100, 4000+100)
ylimExpl = (STAT_ESN_ts_m[6]-0.0115, STAT_ESN_ts_m[6]+0.0115)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlimExpl[0], ylimExpl[0]), xlimExpl[1] - xlimExpl[0], ylimExpl[1] - ylimExpl[0], linewidth=2, edgecolor='blue', facecolor='none',zorder=11)
ax.add_patch(rect)

xlimExpl  = (4000-100, 4000+100)
ylimExpl = (STAT_DR_TS_m[6]-0.0115, STAT_DR_TS_m[6]+0.0115)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlimExpl[0], ylimExpl[0]), xlimExpl[1] - xlimExpl[0], ylimExpl[1] - ylimExpl[0], linewidth=2, edgecolor='blue', facecolor='none',zorder=11)
ax.add_patch(rect)

xlimExpl  = (6545-100, 6545+100)
ylimExpl = (STAT_PR_TS_m-0.0115, STAT_PR_TS_m+0.0115)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlimExpl[0], ylimExpl[0]), xlimExpl[1] - xlimExpl[0], ylimExpl[1] - ylimExpl[0], linewidth=2, edgecolor='blue', facecolor='none',zorder=11)
ax.add_patch(rect)

plt.xlabel('Dimensionality of representations, $D$')
plt.ylabel('NRMSE')
plt.ylim(0.0,0.55)
plt.xlim(1000,10000)
plt.grid(color='0.95')
plt.legend(loc='upper right', fontsize="12")

plt.savefig('/results/Figure_S2_Kuramoto_Sivashinsky.png',bbox_inches="tight")
plt.show() 