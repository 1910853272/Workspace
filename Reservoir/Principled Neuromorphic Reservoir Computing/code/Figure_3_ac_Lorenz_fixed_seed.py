# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in panels a, b, & c of Figure 3.

"""

import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import Lorenz63
from ProductRepresentation import ProductRepresentationRC
from DistributedRepresentation import DistributedRepresentationRC

##
## Data - Lorenz63 system
##
lorenz = Lorenz63()

##
## RC with product representation
##
# ridge parameter for regression
ridge_param = 2.5e-6

productRepresentation = ProductRepresentationRC(system=lorenz, k=2, ridge_param=ridge_param)
productRepresentation.fit12()
productRepresentation.predict12()

print('Product representation, training NRMSE: '+str(productRepresentation.nrmse_train))
print('Product representation, test NRMSE: '+str(productRepresentation.nrmse_test))

##
## RC with distributed representation 
##
# ridge parameter for regression
ridge_param_dr = 1.e-7

distributedRepresentation = DistributedRepresentationRC(system=lorenz, D=28, k=2, ridge_param=ridge_param_dr, seed = 3974999)
distributedRepresentation.fit12()
distributedRepresentation.predict12()

print('Distributed representation, training NRMSE: '+str(distributedRepresentation.nrmse_train))
print('Distributed representation, test NRMSE: '+str(distributedRepresentation.nrmse_test))

##
## Plot
##
# Initial version of this plot comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code

# how much of testtime to plot
plottime=20.
plottime_pts=round(plottime/lorenz.dt)

t_linewidth=1.1
a_linewidth=0.3
plt.rcParams.update({'font.size': 12})

fig1 = plt.figure(dpi=300)
fig1.set_figheight(8)
fig1.set_figwidth(12)

xlabel=[10,15,20,25,30]
h=140
w=150

# top left of grid is 0,0
axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 7), colspan=25, rowspan=36) 
axs2 = plt.subplot2grid(shape=(h,w), loc=(52, 0), colspan=42, rowspan=20)
axs3 = plt.subplot2grid(shape=(h,w), loc=(75, 0), colspan=42, rowspan=20)
axs4 = plt.subplot2grid(shape=(h,w), loc=(98, 0), colspan=42, rowspan=20)

# Product Representration predictions
axs5 = plt.subplot2grid(shape=(h,w), loc=(0, 59), colspan=25, rowspan=36)
axs6 = plt.subplot2grid(shape=(h,w), loc=(52, 50),colspan=42, rowspan=20)
axs7 = plt.subplot2grid(shape=(h,w), loc=(75, 50), colspan=42, rowspan=20)
axs8 = plt.subplot2grid(shape=(h,w), loc=(98, 50), colspan=42, rowspan=20)

# Distributed Representration predictions
axs9 = plt.subplot2grid(shape=(h,w), loc=(0, 109), colspan=25, rowspan=36)
axs10 = plt.subplot2grid(shape=(h,w), loc=(52, 100),colspan=42, rowspan=20)
axs11 = plt.subplot2grid(shape=(h,w), loc=(75, 100), colspan=42, rowspan=20)
axs12 = plt.subplot2grid(shape=(h,w), loc=(98, 100), colspan=42, rowspan=20)

# true Lorenz63 attractor
axs1.plot(lorenz.data.y[0,lorenz.warmtrain_pts:lorenz.maxtime_pts],lorenz.data.y[2,lorenz.warmtrain_pts:lorenz.maxtime_pts],linewidth=a_linewidth, color='k')
axs1.set_xlabel('x', style='italic')
axs1.set_ylabel('z', style='italic')
axs1.set_title('Ground truth')
axs1.text(-.25,.92,'a)', weight = 'bold', ha='left', va='bottom',transform=axs1.transAxes)
axs1.axes.set_xbound(-21,21)
axs1.axes.set_ybound(2,48)

# training phase x
axs2.set_title('training phase') 
axs2.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,lorenz.data.y[0,lorenz.warmup_pts:lorenz.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs2.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,productRepresentation.prediction_train[0,:],linewidth=t_linewidth, color='r', linestyle='dashed')
axs2.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,distributedRepresentation.prediction_train[0,:],linewidth=t_linewidth, color='g', linestyle='dashdot')
axs2.set_ylabel('x', style='italic')
axs2.axes.xaxis.set_ticklabels([])
axs2.axes.set_ybound(-21.,21.)
axs2.axes.set_xbound(-.15,10.15)

# training phase y
axs3.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,lorenz.data.y[1,lorenz.warmup_pts:lorenz.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs3.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,productRepresentation.prediction_train[1,:],linewidth=t_linewidth,color='r', linestyle='dashed')
axs3.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,distributedRepresentation.prediction_train[1,:],linewidth=t_linewidth,color='g', linestyle='dashdot')
axs3.set_ylabel('y', style='italic')
axs3.axes.xaxis.set_ticklabels([])
axs3.axes.set_ybound(-25.,25.)
axs3.axes.set_xbound(-.15,10.15)

# training phase z
axs4.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,lorenz.data.y[2,lorenz.warmup_pts:lorenz.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs4.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,productRepresentation.prediction_train[2,:],linewidth=t_linewidth,color='r', linestyle='dashed')
axs4.plot(lorenz.t_eval[lorenz.warmup_pts:lorenz.warmtrain_pts]-lorenz.warmup,distributedRepresentation.prediction_train[2,:],linewidth=t_linewidth,color='g', linestyle='dashdot')
axs4.set_ylabel('z', style='italic')
axs4.set_xlabel('Time')
axs4.axes.set_xbound(-.15,10.15)

# predicted attractor with Product Representration
axs5.plot(productRepresentation.prediction[0,:],productRepresentation.prediction[2,:],linewidth=a_linewidth,color='r')
axs5.set_xlabel('x', style='italic')
axs5.set_ylabel('z', style='italic')
axs5.set_title('Product; ' + r'$D=28$')
axs5.text(-.25,0.92,'b)', weight = 'bold', ha='left', va='bottom',transform=axs5.transAxes)
axs5.axes.set_xbound(-21,21)
axs5.axes.set_ybound(2,48)

# forecasting phase x
axs6.set_title('forecasting phase')
axs6.set_xticks(xlabel)
axs6.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,lorenz.data.y[0,lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs6.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,productRepresentation.prediction[0,0:plottime_pts],linewidth=t_linewidth,color='r')
axs6.axes.xaxis.set_ticklabels([])
axs6.axes.set_ybound(-21.,21.)
axs6.axes.set_xbound(9.7,30.3)

# forecasting phase y
axs7.set_xticks(xlabel)
axs7.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,lorenz.data.y[1,lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs7.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,productRepresentation.prediction[1,0:plottime_pts],linewidth=t_linewidth,color='r')
#axs7.set_ylabel('y')
axs7.axes.xaxis.set_ticklabels([])
axs7.axes.set_xbound(9.7,30.3)

# forecasting phase z
axs8.set_xticks(xlabel)
axs8.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,lorenz.data.y[2,lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs8.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,productRepresentation.prediction[2,0:plottime_pts],linewidth=t_linewidth,color='r')
axs8.set_xlabel('Time')
axs8.axes.set_xbound(9.7,30.3)

# predicted attractor with Distributed Representration
axs9.plot(distributedRepresentation.prediction[0,:],distributedRepresentation.prediction[2,:],linewidth=a_linewidth,color='g')
axs9.set_xlabel('x', style='italic')
axs9.set_ylabel('z', style='italic')
axs9.set_title('Distributed; ' + r'$D=28$')
axs9.text(-.25,0.92,'c)',weight = 'bold', ha='left', va='bottom',transform=axs9.transAxes)
axs9.axes.set_xbound(-21,21)
axs9.axes.set_ybound(2,48)

# forecasting phase x
axs10.set_title('forecasting phase')
axs10.set_xticks(xlabel)
axs10.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,lorenz.data.y[0,lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs10.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,distributedRepresentation.prediction[0,0:plottime_pts],linewidth=t_linewidth,color='g')
axs10.axes.xaxis.set_ticklabels([])
axs10.axes.set_ybound(-21.,21.)
axs10.axes.set_xbound(9.7,30.3)

# forecasting phase y
axs11.set_xticks(xlabel)
axs11.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,lorenz.data.y[1,lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs11.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,distributedRepresentation.prediction[1,0:plottime_pts],linewidth=t_linewidth,color='g')
axs11.axes.xaxis.set_ticklabels([])
axs11.axes.set_xbound(9.7,30.3)

# forecasting phase z
axs12.set_xticks(xlabel)
axs12.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,lorenz.data.y[2,lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs12.plot(lorenz.t_eval[lorenz.warmtrain_pts-1:lorenz.warmtrain_pts+plottime_pts-1]-lorenz.warmup,distributedRepresentation.prediction[2,0:plottime_pts],linewidth=t_linewidth,color='g')
axs12.set_xlabel('Time')
axs12.axes.set_xbound(9.7,30.3)

fig1.align_ylabels([axs2,axs3,axs4])
plt.savefig('./results/Figure_3_a_c_Lorenz.png')
