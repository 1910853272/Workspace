# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in Figure S.7.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats

simul = 1000
se_mad_coefficient = 1.2533*1.4826/np.sqrt(simul)

# Load the results of experiments for RC with distributed representation, HRR model. As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_DR_HRR = loadmat('/data/Figure_S6_7_8_Lorenz_DR_range_HRR.mat')
Nrange = DATA_DR_HRR["Nrange"][0,:]
# Remove NaNs
STAT_DR_HRR_TS = np.nan_to_num(DATA_DR_HRR["STAT_DR_TS"], nan=1000)
# Get medians
STAT_DR_HRR_TS_m = np.median(STAT_DR_HRR_TS[:,0:simul],axis=1)

# Get intervals
STAT_DR_HRR_TS_err_low  =  STAT_DR_HRR_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_HRR_TS, axis = 1)
STAT_DR_HRR_TS_err_high =  STAT_DR_HRR_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_HRR_TS, axis = 1)

# Load the results of experiments for RC with distributed representation, MAP model. As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_DR_MAP = loadmat('/data/Figure_S7_Lorenz_DR_range_MAP.mat')
# Remove NaNs
STAT_DR_MAP_TS = np.nan_to_num(DATA_DR_MAP["STAT_DR_TS"], nan=1000)
# Get medians
STAT_DR_MAP_TS_m = np.median(STAT_DR_MAP_TS[:,0:simul],axis=1)

# Get intervals
STAT_DR_MAP_TS_err_low  =  STAT_DR_MAP_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_MAP_TS, axis = 1)
STAT_DR_MAP_TS_err_high =  STAT_DR_MAP_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_MAP_TS, axis = 1)

# Load the results of experiments for RC with distributed representation, SBC model. As the experiment involved many runs, the script directly loads the results from the pre-computed files.
DATA_DR_SBC = loadmat('/data/Figure_S7_Lorenz_DR_range_SBC.mat')
Nrange_SBC = DATA_DR_SBC["block_size"][0][0]*DATA_DR_SBC["blockNumRange"][0]+1

# Remove NaNs
STAT_DR_SBC_TS = np.nan_to_num(DATA_DR_SBC["STAT_DR_TS"], nan=1000)
# Get medians
STAT_DR_SBC_TS_m = np.median(STAT_DR_SBC_TS[:,0:simul],axis=1)

# Get intervals
STAT_DR_SBC_TS_err_low  =  STAT_DR_SBC_TS_m - se_mad_coefficient*stats.median_abs_deviation(STAT_DR_SBC_TS, axis = 1)
STAT_DR_SBC_TS_err_high =  STAT_DR_SBC_TS_m + se_mad_coefficient*stats.median_abs_deviation(STAT_DR_SBC_TS, axis = 1)

##
## Plot
##
plt.rcParams.update({'font.size': 13})
plt.figure(figsize=(7.,3.36),dpi=500)
ax = plt.gca()

# Plot the curves
plt.plot(Nrange, STAT_DR_HRR_TS_m ,color ='g', linewidth=2,label= 'HRR')
plt.plot(Nrange, STAT_DR_MAP_TS_m * np.ones((Nrange.size)),color='r', label='MAP', linestyle='dashed',  linewidth=2)
plt.plot(Nrange_SBC, STAT_DR_SBC_TS_m ,color ='k', linewidth=2,label= 'SBC', linestyle='dotted')

# Plot the errors
ax.fill_between(Nrange, STAT_DR_MAP_TS_err_low* np.ones((Nrange.size)), STAT_DR_MAP_TS_err_high* np.ones((Nrange.size)), alpha=0.3,color='r')
ax.fill_between(Nrange, STAT_DR_HRR_TS_err_low, STAT_DR_HRR_TS_err_high, alpha=0.3,color='g')
ax.fill_between(Nrange_SBC, STAT_DR_SBC_TS_err_low, STAT_DR_SBC_TS_err_high, alpha=0.3,color='k')

plt.xlabel('Dimensionality of representations, $D$')
plt.ylabel('NRMSE')
plt.ylim(0.01,1.0)
plt.xlim(10,100)
plt.yscale("log")
plt.grid(color='0.95')
plt.legend(loc='upper right', fontsize="13")
plt.savefig('/results/Figure_S7_Lorenz_VSA_models.png',bbox_inches="tight")
plt.show() 