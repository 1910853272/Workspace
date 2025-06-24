# -*- coding: utf-8 -*-
"""

This script plots the results of experiments reported in panel c of Figure 4.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats

simul = 1000
se_mad_coefficient = 1.2533*1.4826/np.sqrt(simul)

# Load the results of experiments for RC with product representation. As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_PR = loadmat('/data/Figure_4_c_MackeyGlass_ProductRepresentation_3_5.mat')

# Remove NaNs
STAT_PR_TS_3rd = np.nan_to_num(DATA_PR["STAT_PR_TS_3"], nan=100)[:,0:simul]
STAT_PR_TS_4th = np.nan_to_num(DATA_PR["STAT_PR_TS_4"], nan=100)[:,1,0:simul]
STAT_PR_TS_5th = np.nan_to_num(DATA_PR["STAT_PR_TS_5"], nan=100)[:,5,0:simul]

# Get medians
STAT_PR_TS_3rd_m = np.median(STAT_PR_TS_3rd)
STAT_PR_TS_4th_m = np.median(STAT_PR_TS_4th)
STAT_PR_TS_5th_m = np.median(STAT_PR_TS_5th)

# Get intervals
STAT_PR_TS_3rd_err_low  = STAT_PR_TS_3rd_m - se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS_3rd, axis=1)
STAT_PR_TS_3rd_err_high = STAT_PR_TS_3rd_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS_3rd, axis=1)

STAT_PR_TS_4th_err_low  = STAT_PR_TS_4th_m - se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS_4th, axis=1)
STAT_PR_TS_4th_err_high = STAT_PR_TS_4th_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS_4th, axis=1)

STAT_PR_TS_5th_err_low  = STAT_PR_TS_5th_m - se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS_5th, axis=1)
STAT_PR_TS_5th_err_high = STAT_PR_TS_5th_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_PR_TS_5th, axis=1)

# Load the results of experiments for RC with distributed representation 
DATA_DR = loadmat('/data/Figure_4_c_MackeyGlass_DistributedRepresentation_4_7.mat')

Ridgerange = DATA_DR["Ridgerange"][0,:]
Nrange = DATA_DR["Nrange"][0,:]

ind_ridge_4 = 5
ind_ridge_5 = 4
ind_ridge_6 = 4
ind_ridge_7 = 5

# Remove NaNs
STAT_DR_TS_4 = np.nan_to_num(DATA_DR["STAT_DR_TS_4"], nan=100)
STAT_DR_TS_5 = np.nan_to_num(DATA_DR["STAT_DR_TS_5"], nan=100)
STAT_DR_TS_6 = np.nan_to_num(DATA_DR["STAT_DR_TS_6"], nan=100)
STAT_DR_TS_7 = np.nan_to_num(DATA_DR["STAT_DR_TS_7"], nan=100)

# Get medians
STAT_DR_TS_4_m = np.median(STAT_DR_TS_4[:,:,0:simul],axis=2)
STAT_DR_TS_5_m = np.median(STAT_DR_TS_5[:,:,0:simul],axis=2)
STAT_DR_TS_6_m = np.median(STAT_DR_TS_6[:,:,0:simul],axis=2)
STAT_DR_TS_7_m = np.median(STAT_DR_TS_7[:,:,0:simul],axis=2)

# Get intervals
STAT_DR_TS_4_err_low  =  STAT_DR_TS_4_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS_4, axis = 2)
STAT_DR_TS_4_err_high =  STAT_DR_TS_4_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS_4, axis = 2)

STAT_DR_TS_5_err_low  =  STAT_DR_TS_5_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS_5, axis = 2)
STAT_DR_TS_5_err_high =  STAT_DR_TS_5_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS_5, axis = 2)

STAT_DR_TS_6_err_low  =  STAT_DR_TS_6_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS_6, axis = 2)
STAT_DR_TS_6_err_high =  STAT_DR_TS_6_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS_6, axis = 2)

STAT_DR_TS_7_err_low  =  STAT_DR_TS_7_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS_7, axis = 2)
STAT_DR_TS_7_err_high =  STAT_DR_TS_7_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_TS_7, axis = 2)

# Load the results of experiments for the MLPs
DATA_mlp_84= loadmat('/data/Figure_4_c_MackeyGlass_MLPRegressor_84.mat')
DATA_mlp_210= loadmat('/data/Figure_4_c_MackeyGlass_MLPRegressor_210.mat')
DATA_mlp_462= loadmat('/data/Figure_4_c_MackeyGlass_MLPRegressor_462.mat')

# Remove NaNs
STAT_MLP_84_ts = np.nan_to_num(DATA_mlp_84["STAT_MLP_TS"], nan=100)[:,0,0:simul]
STAT_MLP_210_ts = np.nan_to_num(DATA_mlp_210["STAT_MLP_TS"], nan=100)[:,0,0:simul]
STAT_MLP_462_ts = np.nan_to_num(DATA_mlp_462["STAT_MLP_TS"], nan=100)[:,0,0:simul]

# Get medians
STAT_MLP_84_ts_m = np.median(STAT_MLP_84_ts)
STAT_MLP_210_ts_m = np.median(STAT_MLP_210_ts)
STAT_MLP_462_ts_m = np.median(STAT_MLP_462_ts)

# Get intervals
STAT_MLP_84_ts_err_low  = STAT_MLP_84_ts_m - se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_84_ts, axis=1)
STAT_MLP_84_ts_err_high = STAT_MLP_84_ts_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_84_ts, axis=1)

STAT_MLP_210_ts_err_low  = STAT_MLP_210_ts_m - se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_210_ts, axis=1)
STAT_MLP_210_ts_err_high = STAT_MLP_210_ts_m +  se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_210_ts, axis=1)

STAT_MLP_462_ts_err_low  = STAT_MLP_462_ts_m - se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_462_ts, axis=1)
STAT_MLP_462_ts_err_high = STAT_MLP_462_ts_m + se_mad_coefficient*stats.median_abs_deviation(STAT_MLP_462_ts, axis=1)

# Load the results of experiments for the ESNs
DATA_esn = loadmat('/data/Figure_4_c_MackeyGlass_Echo_State_Network.mat')
Nrange_esn = DATA_esn["Nrange"][0,:]

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
scale = 1.3
plt.figure(figsize=(5.,3.36),dpi=500)
ax = plt.gca()

# Plot the curves
ax.plot(Nrange, STAT_DR_TS_4_m[:,ind_ridge_4] ,color ='g', linewidth=1.5,linestyle='dashed', label= 'Distributed, 4th')
ax.plot(Nrange, STAT_DR_TS_5_m[:,ind_ridge_5] ,color ='g', linewidth=1.5, linestyle='dashdot', label= 'Distributed, 5th')
ax.plot(Nrange, STAT_DR_TS_6_m[:,ind_ridge_6] ,color ='g', linewidth=1.5, linestyle='dotted', label= 'Distributed, 6th')
ax.plot(Nrange, STAT_DR_TS_7_m[:,ind_ridge_7] ,color ='g', linewidth=2.5, linestyle='solid', label= 'Distributed, 7th')

ax.plot(Nrange, STAT_PR_TS_3rd_m * np.ones((Nrange.size)),color='r', label='Product, 3rd - $D=$84', linestyle='solid',  linewidth=1.)
ax.plot(Nrange, STAT_PR_TS_4th_m * np.ones((Nrange.size)),color='r', label='Product, 4th - $D=$210', linestyle='dashed',  linewidth=1.5) # Picked optimal ridge value for NGRC
ax.plot(Nrange, STAT_PR_TS_5th_m * np.ones((Nrange.size)),color='r', label='Product, 5th - $D=$462', linestyle='dashdot',  linewidth=1.5) # Picked optimal ridge value for NGRC

ax.plot(Nrange_esn, STAT_ESN_ts_m ,color ='k', linewidth=1.5,label= 'Echo State Network', linestyle='dotted')

ax.plot(Nrange, STAT_MLP_84_ts_m * np.ones((Nrange.size)),color='b', label='Perceptron - (84, 256, 256, 32)', linestyle='solid',  linewidth=1.5) # Picked optimal ridge value for NGRC
ax.plot(Nrange, STAT_MLP_210_ts_m * np.ones((Nrange.size)),color='b', label='Perceptron - (210, 256, 256, 32)', linestyle='dashed',  linewidth=1.5) # Picked optimal ridge value for NGRC
ax.plot(Nrange, STAT_MLP_462_ts_m * np.ones((Nrange.size)),color='b', label='Perceptron - (462, 256, 256, 32)', linestyle='dashdot',  linewidth=1.5) # Picked optimal ridge value for NGRC

# Plot the errors
ax.fill_between(Nrange, STAT_DR_TS_4_err_low[:,ind_ridge_4], STAT_DR_TS_4_err_high[:,ind_ridge_4], alpha=0.3,color='g')
ax.fill_between(Nrange, STAT_DR_TS_5_err_low[:,ind_ridge_5], STAT_DR_TS_5_err_high[:,ind_ridge_5], alpha=0.3,color='g')
ax.fill_between(Nrange, STAT_DR_TS_6_err_low[:,ind_ridge_6], STAT_DR_TS_6_err_high[:,ind_ridge_6], alpha=0.3,color='g')
ax.fill_between(Nrange, STAT_DR_TS_7_err_low[:,ind_ridge_7], STAT_DR_TS_7_err_high[:,ind_ridge_7], alpha=0.3,color='g')

ax.fill_between(Nrange, STAT_PR_TS_3rd_err_low* np.ones((Nrange.size)), STAT_PR_TS_3rd_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')
ax.fill_between(Nrange, STAT_PR_TS_4th_err_low* np.ones((Nrange.size)), STAT_PR_TS_4th_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')
ax.fill_between(Nrange, STAT_PR_TS_5th_err_low* np.ones((Nrange.size)), STAT_PR_TS_5th_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')

ax.fill_between(Nrange_esn, STAT_ESN_ts_err_low, STAT_ESN_ts_err_high, alpha=0.3,color='k')

ax.fill_between(Nrange, STAT_MLP_84_ts_err_low* np.ones((Nrange.size)), STAT_MLP_84_ts_err_high* np.ones((Nrange.size)), alpha=0.3,color='b')
ax.fill_between(Nrange, STAT_MLP_210_ts_err_low* np.ones((Nrange.size)), STAT_MLP_210_ts_err_high* np.ones((Nrange.size)), alpha=0.3,color='b')
ax.fill_between(Nrange, STAT_MLP_462_ts_err_high* np.ones((Nrange.size)), STAT_MLP_462_ts_m* np.ones((Nrange.size)), alpha=0.3,color='b')

# Emphasize the interesting configurations
# RC PR configuration
xlim3rd  = (84-4.6, 84+4.6)
ylim3rd = (STAT_PR_TS_3rd_m-0.047, STAT_PR_TS_3rd_m+0.047)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlim3rd[0], ylim3rd[0]), xlim3rd[1] - xlim3rd[0], ylim3rd[1] - ylim3rd[0], linewidth=1, edgecolor='r', facecolor='r',zorder=10)
ax.add_patch(rect)

# RC PR configuration
xlim4th  = (210-4.6, 210+4.6)
ylim4th = (STAT_PR_TS_4th_m-0.015, STAT_PR_TS_4th_m+0.015)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlim4th[0], ylim4th[0]), xlim4th[1] - xlim4th[0], ylim4th[1] - ylim4th[0], linewidth=1, edgecolor='r', facecolor='r',zorder=10)
ax.add_patch(rect)

# RC PR configuration
xlim5th  = (462-4.6, 462+4.6)
ylim5th = (STAT_PR_TS_5th_m-0.012, STAT_PR_TS_5th_m+0.013)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlim5th[0], ylim5th[0]), xlim5th[1] - xlim5th[0], ylim5th[1] - ylim5th[0], linewidth=1, edgecolor='r', facecolor='r',zorder=10)
ax.add_patch(rect)

# RC DR configuration
# draw the zoomed ellipse
ell = mpl.patches.Ellipse((100, STAT_DR_TS_4_m[3,ind_ridge_4]), 3*3.6, 3.2*0.013, linewidth=1, edgecolor='g', facecolor='g',zorder=10)
ax.add_patch(ell)

# MLP configurations
xlim84  = (84-4.6, 84+4.6)
ylim84 = (STAT_MLP_84_ts_m-0.07, STAT_MLP_84_ts_m+0.11)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlim84[0], ylim84[0]), xlim84[1] - xlim84[0], ylim84[1] - ylim84[0], linewidth=1, edgecolor='b', facecolor='b',zorder=10)
ax.add_patch(rect)

xlim210  = (210-4.6, 210+4.6)
ylim210 = (STAT_MLP_210_ts_m-0.07, STAT_MLP_210_ts_m+0.11)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlim210[0], ylim210[0]), xlim210[1] - xlim210[0], ylim210[1] - ylim210[0], linewidth=1, edgecolor='b', facecolor='b',zorder=10)
ax.add_patch(rect)

xlim462  = (462-4.6, 462+4.6)
ylim462 = (STAT_MLP_462_ts_m-0.07, STAT_MLP_462_ts_m+0.11)
# draw the zoomed rectangle
rect = mpl.patches.Rectangle((xlim462[0], ylim462[0]), xlim462[1] - xlim462[0], ylim462[1] - ylim462[0], linewidth=1, edgecolor='b', facecolor='b',zorder=1)
ax.add_patch(rect)

plt.xlabel('Dimensionality of representations, '+ r'$D$')
plt.ylabel('NRMSE')
plt.ylim(0.02,90.0)
plt.yscale("log")
plt.xlim(50.0,500.)

plt.grid(color='0.95')
ax.legend(loc='upper right', fontsize="8")
ax.set_zorder(1)
plt.text(-9.,175.95,'c)', weight='bold', ha='left', va='top')
plt.title('Mackey-Glass')

plt.savefig('/results/Figure_4_c_MackeyGlass.png',bbox_inches="tight")
plt.show() 