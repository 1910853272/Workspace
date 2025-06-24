# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in Figure S.4.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the results of experiments for RC with distributed representation for random Sigma-Pi networks. As the experiment involved many runs, the script directly loads the results from the pre-computed files. 
DATA = loadmat('/data/Figure_S4_MackeyGlass_radnom_SigmaPy_3-7th.mat')

STAT_DR_TS_3 = np.nan_to_num(DATA["STAT_DR_TS_3"], nan=100)
STAT_DR_TS_3_m = np.median(STAT_DR_TS_3[:,4,0:],axis=1)

STAT_DR_TS_4 = np.nan_to_num(DATA["STAT_DR_TS_4"], nan=100)
STAT_DR_TS_4_m = np.median(STAT_DR_TS_4[:,4,0:],axis=1)

STAT_DR_TS_5 = np.nan_to_num(DATA["STAT_DR_TS_5"], nan=100)
STAT_DR_TS_5_m = np.median(STAT_DR_TS_5[:,5,0:],axis=1)

STAT_DR_TS_6 = np.nan_to_num(DATA["STAT_DR_TS_6"], nan=100)
STAT_DR_TS_6_m = np.median(STAT_DR_TS_6[:,4,0:],axis=1)

STAT_DR_TS_7 = np.nan_to_num(DATA["STAT_DR_TS_7"], nan=100)
STAT_DR_TS_7_m = np.median(STAT_DR_TS_7[:,4,0:],axis=1)

Ridgerange = DATA["Ridgerange"][0,:]
Nrange = DATA["Nrange"][0,:]

# Load the results of experiments for RC with product representation. As the experiment involved many runs, the script directly loads the results from the pre-computed files. 
DATA_PR = loadmat('/data/Figure_4_c_MackeyGlass_ProductRepresentation_3_5.mat')

STAT_PR_TS_3rd_m = np.median(np.nan_to_num(DATA_PR["STAT_PR_TS_3"], nan=100)[:,0:])
STAT_PR_TS_4th_m = np.median(np.nan_to_num(DATA_PR["STAT_PR_TS_4"], nan=100)[:,1,0:])
STAT_PR_TS_5th_m = np.median(np.nan_to_num(DATA_PR["STAT_PR_TS_5"], nan=100)[:,5,0:])

# Load the results of experiments for RC with distributed representation. As the experiment involved many runs, the script directly loads the results from the pre-computed files. 
DATA_DR = loadmat('/data/Figure_4_c_MackeyGlass_DistributedRepresentation_4_7.mat')

STAT_DR_TS_5_structured = np.nan_to_num(DATA_DR["STAT_DR_TS_5"], nan=100)
STAT_DR_TS_5_structured_m = np.median(STAT_DR_TS_5_structured[:,4,0:],axis=1)
Nrange_structured = DATA_DR["Nrange"][0,:]
 
##
## Plot
##

plt.figure(figsize=(6.,3.),dpi=600)
ax = plt.gca()

# RC DR configurations with random Sigma-Pi networks
ax.plot(Nrange, STAT_DR_TS_3_m ,color ='g', linewidth=2.50,linestyle='solid', label= 'Distributed, 3rd')
ax.plot(Nrange, STAT_DR_TS_4_m ,color ='g', linewidth=1.5,linestyle='dashed', label= 'Distributed, 4th')
ax.plot(Nrange, STAT_DR_TS_5_m ,color ='g', linewidth=1.5, linestyle='dashdot', label= 'Distributed, 5th')
ax.plot(Nrange, STAT_DR_TS_6_m ,color ='g', linewidth=1.5, linestyle='dotted', label= 'Distributed, 6th')
ax.plot(Nrange, STAT_DR_TS_7_m ,color ='g', linewidth=1.0, linestyle='solid', label= 'Distributed, 7th')

# RC PR configurations
ax.plot(Nrange, STAT_PR_TS_3rd_m * np.ones((Nrange.size)),color='r', label='Product, 3rd - $D=$84', linestyle='solid',  linewidth=1.)
ax.plot(Nrange, STAT_PR_TS_4th_m * np.ones((Nrange.size)),color='r', label='Product, 4th - $D=$210', linestyle='dashed',  linewidth=1.) 
ax.plot(Nrange, STAT_PR_TS_5th_m * np.ones((Nrange.size)),color='r', label='Product, 5th - $D=$462', linestyle='dashdot',  linewidth=1.) 

# RC DR configurations
ax.plot(Nrange_structured, STAT_DR_TS_5_structured_m,color = '0.8', linewidth=1.5, linestyle='dashdot')

# RC DR configurations
xlim3rd  = (84-4.6, 84+4.6)
ylim3rd = (STAT_PR_TS_3rd_m-0.015, STAT_PR_TS_3rd_m+0.015)
# draw the zoomed rectangle on the whole
rect = mpl.patches.Rectangle((xlim3rd[0], ylim3rd[0]), xlim3rd[1] - xlim3rd[0], ylim3rd[1] - ylim3rd[0], linewidth=1, edgecolor='k', facecolor='none',zorder=10)
ax.add_patch(rect)

xlim4th  = (210-4.6, 210+4.6)
ylim4th = (STAT_PR_TS_4th_m-0.015, STAT_PR_TS_4th_m+0.015)
# draw the zoomed rectangle on the whole
rect2 = mpl.patches.Rectangle((xlim4th[0], ylim4th[0]), xlim4th[1] - xlim4th[0], ylim4th[1] - ylim4th[0], linewidth=1, edgecolor='k', facecolor='none',zorder=10)
ax.add_patch(rect2)

xlim5th  = (462-4.6, 462+4.6)
ylim5th = (STAT_PR_TS_5th_m-0.015, STAT_PR_TS_5th_m+0.015)
# draw the zoomed rectangle on the whole
rect2 = mpl.patches.Rectangle((xlim5th[0], ylim5th[0]), xlim5th[1] - xlim5th[0], ylim5th[1] - ylim5th[0], linewidth=1, edgecolor='k', facecolor='none',zorder=10)
ax.add_patch(rect2)

plt.xlabel('Dimensionality of representations, $D$')
plt.ylabel('NRMSE')
plt.ylim(0.0,0.8)
plt.xlim(50.0,500.)

plt.grid(color='0.95')
plt.legend(loc='upper right', prop={'size': 10.5})

plt.savefig('/results/Figure_S4_MackeyGlass_radnom_SigmaPy.png',bbox_inches="tight")
plt.show() 