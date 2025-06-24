# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in Figure S.6.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats

simul = 1000
se_mad_coefficient = 1.2533*1.4826/np.sqrt(simul)

# Load the results of experiments for RC with distributed representation, 1st-order.  As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_DR_1st = loadmat('/data/Figure_S6_Lorenz_DR_range_1st_order.mat')
# Remove NaNs
STAT_DR_1st_TS = np.nan_to_num(DATA_DR_1st["STAT_DR_TS"], nan=1000)
# Get medians
STAT_DR_1st_TS_m = np.median(STAT_DR_1st_TS[:,0:simul],axis=1)

# Get intervals
STAT_DR_1st_TS_err_low  =  STAT_DR_1st_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_1st_TS, axis = 1)
STAT_DR_1st_TS_err_high =  STAT_DR_1st_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_1st_TS, axis = 1)

# Load the results of experiments for RC with distributed representation, 1st- & 2nd- orders.  As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_DR_HRR = loadmat('/data/Figure_S6_7_8_Lorenz_DR_range_HRR.mat')
Nrange = DATA_DR_HRR["Nrange"][0,:]
# Remove NaNs
STAT_DR_HRR_TS = np.nan_to_num(DATA_DR_HRR["STAT_DR_TS"], nan=1000)
# Get medians
STAT_DR_HRR_TS_m = np.median(STAT_DR_HRR_TS[:,0:simul],axis=1)

# Get intervals
STAT_DR_HRR_TS_err_low  =  STAT_DR_HRR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_HRR_TS, axis = 1)
STAT_DR_HRR_TS_err_high =  STAT_DR_HRR_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_HRR_TS, axis = 1)

# Load the results of experiments for RC with distributed representation, 1st-, 2nd- , & 3rd- orders.  As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_DR_1st2nd3rd = loadmat('/data/Figure_S6_Lorenz_DR_range_1st_2nd_3rd_order.mat')
# Remove NaNs
STAT_DR_1st2nd3rd_TS = np.nan_to_num(DATA_DR_1st2nd3rd["STAT_DR_TS"], nan=1000)
# Get medians
STAT_DR_1st2nd3rd_TS_m = np.median(STAT_DR_1st2nd3rd_TS[:,0:simul],axis=1)

# Get intervals
STAT_DR_1st2nd3rd_TS_err_low  =  STAT_DR_1st2nd3rd_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_1st2nd3rd_TS, axis = 1)
STAT_DR_1st2nd3rd_TS_err_high =  STAT_DR_1st2nd3rd_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_1st2nd3rd_TS, axis = 1)

##
## Plot
##
plt.rcParams.update({'font.size': 13})
plt.figure(figsize=(7.,3.36),dpi=500)
ax = plt.gca()

# Plot the curves
plt.plot(Nrange, STAT_DR_1st_TS_m * np.ones((Nrange.size)),color='r', label='1st-order', linestyle='dashed',  linewidth=2)
plt.plot(Nrange, STAT_DR_HRR_TS_m ,color ='g', linewidth=2,label= '1st- & 2nd- orders')
plt.plot(Nrange, STAT_DR_1st2nd3rd_TS_m ,color ='k', linewidth=2,label= '1st-, 2nd- , & 3rd- orders', linestyle='dotted')

# Plot the errors
ax.fill_between(Nrange, STAT_DR_1st_TS_err_low* np.ones((Nrange.size)), STAT_DR_1st_TS_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')
ax.fill_between(Nrange, STAT_DR_HRR_TS_err_low, STAT_DR_HRR_TS_err_high, alpha=0.3,color='g')
ax.fill_between(Nrange, STAT_DR_1st2nd3rd_TS_err_low, STAT_DR_1st2nd3rd_TS_err_high, alpha=0.3,color='k')

plt.xlabel('Dimensionality of representations, $D$')
plt.ylabel('NRMSE')
plt.ylim(0.01,1.0)
plt.xlim(10,100)
plt.yscale("log")
plt.grid(color='0.95')
plt.legend(loc='center right', fontsize="13")
plt.savefig('/results/Figure_S6_Lorenz_ablation_orders.png',bbox_inches="tight")
plt.show() 