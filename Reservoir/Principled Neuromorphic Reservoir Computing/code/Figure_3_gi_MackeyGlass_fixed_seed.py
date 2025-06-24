# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in panels g, h, & i of Figure 3.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from DataGeneration import MackeyGlass
from ProductRepresentation import ProductRepresentationRC
from DistributedRepresentation import DistributedRepresentationRC

##
## Data - Mackey-Glass system
##
mackeyGlass = MackeyGlass()    
taps = np.array([0,1, np.floor(mackeyGlass.tau/(2*mackeyGlass.dt)),np.floor(mackeyGlass.tau/(2*mackeyGlass.dt))+1,np.floor(mackeyGlass.tau/mackeyGlass.dt),np.floor(mackeyGlass.tau/mackeyGlass.dt)+1], dtype=int) 

##
## RC with product representation
##

# ridge parameter for regression
ridge_param = 1.e-7

productRepresentation = ProductRepresentationRC(system=mackeyGlass, k=6, taps=taps, ridge_param=ridge_param)
productRepresentation.fit123()
productRepresentation.predict123()

print('Product representation, training NRMSE: '+str(productRepresentation.nrmse_train))
print('Product representation, test NRMSE: '+str(productRepresentation.nrmse_test))

##
## RC with distributed representation 
##

# ridge parameter for regression
ridge_param_dr = 1.e-6

distributedRepresentation = DistributedRepresentationRC(system=mackeyGlass, D=100, k=6, taps=taps, ridge_param=ridge_param_dr, seed = 10, normalize = True)
distributedRepresentation.fit1234()
distributedRepresentation.predict1234()

print('Distributed representation, training NRMSE: '+str(distributedRepresentation.nrmse_train))
print('Distributed representation, test NRMSE: '+str(distributedRepresentation.nrmse_test))



##
## Plot
##
# Amount of time to plot prediction for
plottime=1800.
plottime_pts=round(plottime/mackeyGlass.dt)

pts_tau = np.floor(mackeyGlass.tau/mackeyGlass.dt).astype(int)
t_linewidth=1.1
a_linewidth=0.3
xlabel=np.arange(0, mackeyGlass.traintime+0.01, 400)
xlabel_ts=np.arange(mackeyGlass.traintime, mackeyGlass.traintime+plottime+0.01, 400)

plt.rcParams.update({'font.size': 12})

fig1 = plt.figure(dpi=300)
fig1.set_figheight(8)
fig1.set_figwidth(12)

h=140
w=150

# top left of grid is 0,0
axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 7), colspan=25, rowspan=36) 
axs2 = plt.subplot2grid(shape=(h,w), loc=(52, 0), colspan=42, rowspan=20)

axs3 = plt.subplot2grid(shape=(h,w), loc=(0, 59), colspan=25, rowspan=36)
axs4 = plt.subplot2grid(shape=(h,w), loc=(52, 50),colspan=42, rowspan=20)

# VFA predictions
axs5 = plt.subplot2grid(shape=(h,w), loc=(0, 109), colspan=25, rowspan=36)
axs6 = plt.subplot2grid(shape=(h,w), loc=(52, 100),colspan=42, rowspan=20)


# true Lorenz attractor
axs1.plot(mackeyGlass.data.y[0,pts_tau:],mackeyGlass.data.y[0,0:mackeyGlass.maxtime_pts-pts_tau],linewidth=a_linewidth, color='k')
axs1.set_xlabel('u(t)', style='italic')
axs1.set_ylabel('u(t-17)', style='italic')
axs1.set_title('Ground truth')
axs1.text(-.25,.92,'g)', weight='bold', ha='left', va='bottom',transform=axs1.transAxes)
axs1.axes.set_xbound(.3,1.4)
axs1.axes.set_ybound(.3,1.4)



# training phase x
axs2.set_title('training phase') 
axs2.plot(mackeyGlass.t_eval[mackeyGlass.warmup_pts:mackeyGlass.warmtrain_pts]-mackeyGlass.warmup,mackeyGlass.data.y[0,mackeyGlass.warmup_pts:mackeyGlass.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs2.plot(mackeyGlass.t_eval[mackeyGlass.warmup_pts:mackeyGlass.warmtrain_pts]-mackeyGlass.warmup,productRepresentation.prediction_train[0,:],linewidth=t_linewidth, color='r', linestyle='dashed')
axs2.plot(mackeyGlass.t_eval[mackeyGlass.warmup_pts:mackeyGlass.warmtrain_pts]-mackeyGlass.warmup,distributedRepresentation.prediction_train[0,:],linewidth=t_linewidth, color='g', linestyle='dashdot')
axs2.set_ylabel('u', style='italic')
axs2.axes.set_ybound(.3,1.4)
axs2.axes.set_xbound(0,mackeyGlass.traintime)
axs2.set_xticks(xlabel)
axs2.set_xlabel('Time')





# prediction attractor
axs3.plot(productRepresentation.prediction[0,pts_tau:],productRepresentation.prediction[0,0:mackeyGlass.testtime_pts-pts_tau],linewidth=a_linewidth,color='r')
axs3.set_xlabel('u(t)', style='italic')
axs3.set_ylabel('u(t-17)', style='italic')
axs3.set_title('Product; ' + r'$D=84$')
axs3.text(-.25,0.92,'h)', weight='bold',ha='left', va='bottom',transform=axs3.transAxes)
axs3.axes.set_xbound(.3,1.4)
axs3.axes.set_ybound(.3,1.4)

# forecasting phase x
axs4.set_title('forecasting phase')
axs4.plot(mackeyGlass.t_eval[mackeyGlass.warmtrain_pts-1:mackeyGlass.warmtrain_pts+plottime_pts-1]-mackeyGlass.warmup,mackeyGlass.data.y[0,mackeyGlass.warmtrain_pts-1:mackeyGlass.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs4.plot(mackeyGlass.t_eval[mackeyGlass.warmtrain_pts-1:mackeyGlass.warmtrain_pts+plottime_pts-1]-mackeyGlass.warmup,productRepresentation.prediction[0,0:plottime_pts],linewidth=t_linewidth,color='r')
axs4.text(-.155*1.15,0.87,'d)', ha='left', va='bottom',transform=axs4.transAxes)
axs4.axes.set_xbound(mackeyGlass.traintime,plottime+mackeyGlass.traintime)
axs4.axes.set_ybound(.3,1.4)
axs4.set_xlabel('Time')
axs4.set_xticks(xlabel_ts)


# prediction attractor VFA

# prediction attractor
axs5.plot(distributedRepresentation.prediction[0,pts_tau:],distributedRepresentation.prediction[0,0:mackeyGlass.testtime_pts-pts_tau],linewidth=a_linewidth,color='g')
axs5.set_xlabel('u(t)', style='italic')
axs5.set_ylabel('u(t-17)', style='italic')
axs5.set_title('Distributed; ' + r'$D=100$')
axs5.text(-.25,0.92,'i)', weight='bold', ha='left', va='bottom',transform=axs5.transAxes)
axs5.axes.set_xbound(.3,1.4)
axs5.axes.set_ybound(.3,1.4)


# forecasting phase x
axs6.set_title('forecasting phase')
axs6.plot(mackeyGlass.t_eval[mackeyGlass.warmtrain_pts-1:mackeyGlass.warmtrain_pts+plottime_pts-1]-mackeyGlass.warmup,mackeyGlass.data.y[0,mackeyGlass.warmtrain_pts-1:mackeyGlass.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs6.plot(mackeyGlass.t_eval[mackeyGlass.warmtrain_pts-1:mackeyGlass.warmtrain_pts+plottime_pts-1]-mackeyGlass.warmup,distributedRepresentation.prediction[0,0:plottime_pts],linewidth=t_linewidth,color='g')
axs6.axes.set_xbound(mackeyGlass.traintime,plottime+mackeyGlass.traintime)
axs6.axes.set_ybound(.3,1.4)
axs6.set_xlabel('Time')
axs6.set_xticks(xlabel_ts)

plt.savefig('/results/Figure_3_gi_MackeyGlass.png')
plt.show() 