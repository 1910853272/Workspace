# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in panels d, e, & f of Figure 3.

"""

import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import DoubleScroll
from ProductRepresentation import ProductRepresentationRC
from DistributedRepresentation import DistributedRepresentationRC

##
## Data - Double-Scroll system
##
doubleScroll = DoubleScroll()    
    
##
## RC with product representation
##
# ridge parameter for regression
ridge_param = 1.e-3

productRepresentation = ProductRepresentationRC(system=doubleScroll, k=2, ridge_param=ridge_param, bias = False)
productRepresentation.fit13()
productRepresentation.predict13()

print('Product representation, training NRMSE: '+str(productRepresentation.nrmse_train))
print('Product representation, test NRMSE: '+str(productRepresentation.nrmse_test))

##
## RC with distributed representation 
##
# ridge parameter for regression
ridge_param_dr = 1.e-5

distributedRepresentation = DistributedRepresentationRC(system=doubleScroll, D=34, k=2, ridge_param=ridge_param_dr, seed = 5)
distributedRepresentation.fit13()
distributedRepresentation.predict13()

print('Distributed representation, training NRMSE: '+str(distributedRepresentation.nrmse_train))
print('Distributed representation, test NRMSE: '+str(distributedRepresentation.nrmse_test))

##
## Plot
##
# Initial version of this plot comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code

# Amount of time to plot prediction for
plottime=200.
plottime_pts=round(plottime/doubleScroll.dt)

t_linewidth=1.1
a_linewidth=0.3
plt.rcParams.update({'font.size': 12})

fig1 = plt.figure(dpi=300)
fig1.set_figheight(8)
fig1.set_figwidth(12)

xlabel=[100,150,200,250,300]
sectionskip = 2
h=140 + sectionskip
w=150

# top left of grid is 0,0
axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 7), colspan=25, rowspan=36) 
axs2 = plt.subplot2grid(shape=(h,w), loc=(52+sectionskip, 0), colspan=42, rowspan=20)
axs3 = plt.subplot2grid(shape=(h,w), loc=(75+sectionskip, 0), colspan=42, rowspan=20)
axs4 = plt.subplot2grid(shape=(h,w), loc=(98+sectionskip, 0), colspan=42, rowspan=20)

# Product Representration predictions
axs5 = plt.subplot2grid(shape=(h,w), loc=(0, 59), colspan=25, rowspan=36)
axs6 = plt.subplot2grid(shape=(h,w), loc=(52+sectionskip, 50),colspan=42, rowspan=20)
axs7 = plt.subplot2grid(shape=(h,w), loc=(75+sectionskip, 50), colspan=42, rowspan=20)
axs8 = plt.subplot2grid(shape=(h,w), loc=(98+sectionskip, 50), colspan=42, rowspan=20)

# Distributed Representration predictions
axs9 = plt.subplot2grid(shape=(h,w), loc=(0, 109), colspan=25, rowspan=36)
axs10 = plt.subplot2grid(shape=(h,w), loc=(52+sectionskip, 100),colspan=42, rowspan=20)
axs11 = plt.subplot2grid(shape=(h,w), loc=(75+sectionskip, 100), colspan=42, rowspan=20)
axs12 = plt.subplot2grid(shape=(h,w), loc=(98+sectionskip, 100), colspan=42, rowspan=20)

# true Double-Scroll attractor
axs1.plot(doubleScroll.data.y[0,doubleScroll.warmtrain_pts:doubleScroll.maxtime_pts],doubleScroll.data.y[2,doubleScroll.warmtrain_pts:doubleScroll.maxtime_pts],linewidth=a_linewidth, color='k')
axs1.set_xlabel('$V_1$')
axs1.set_ylabel('$I$')
axs1.set_title('Ground truth')
axs1.text(-.35,.92,'d)', weight='bold', ha='left', va='bottom',transform=axs1.transAxes)
axs1.axes.set_xbound(-2.1, 2.1)
axs1.axes.set_ybound(-2.5, 2.5)

# training phase V_1
axs2.set_title('training phase')
axs2.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,doubleScroll.data.y[0,doubleScroll.warmup_pts:doubleScroll.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs2.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,productRepresentation.prediction_train[0,:],linewidth=t_linewidth, color='r', linestyle='dashed')
axs2.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,distributedRepresentation.prediction_train[0,:],linewidth=t_linewidth, color='g', linestyle='dashdot')
axs2.set_ylabel('$V_1$')
axs2.axes.xaxis.set_ticklabels([])
axs2.axes.set_xbound(-1.5,101.5)
axs2.axes.set_ybound(-2.1, 2.1)

# training phase V_2
axs3.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,doubleScroll.data.y[1,doubleScroll.warmup_pts:doubleScroll.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs3.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,productRepresentation.prediction_train[1,:],linewidth=t_linewidth,color='r', linestyle='dashed')
axs3.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,distributedRepresentation.prediction_train[1,:],linewidth=t_linewidth,color='g', linestyle='dashdot')
axs3.set_ylabel('$V_2$')
axs3.axes.xaxis.set_ticklabels([])
axs3.axes.set_xbound(-1.5,101.5)
axs3.axes.set_ybound(-1.1, 1.1)

# training phase I
axs4.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,doubleScroll.data.y[2,doubleScroll.warmup_pts:doubleScroll.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs4.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,productRepresentation.prediction_train[2,:],linewidth=t_linewidth,color='r', linestyle='dashed')
axs4.plot(doubleScroll.t_eval[doubleScroll.warmup_pts:doubleScroll.warmtrain_pts]-doubleScroll.warmup,distributedRepresentation.prediction_train[2,:],linewidth=t_linewidth,color='g', linestyle='dashdot')
axs4.set_ylabel('$I$')
axs4.set_xlabel('Time')
axs4.axes.set_xbound(-1.5,101.5)
axs4.axes.set_ybound(-2.5, 2.5)

# predicted attractor with Product Representration
axs5.plot(productRepresentation.prediction[0,:],productRepresentation.prediction[2,:],linewidth=a_linewidth,color='r')
axs5.set_xlabel('$V_1$')
axs5.set_ylabel('$I$')
axs5.set_title('Product; ' + r'$D=62$')
axs5.text(-.35,0.92,'e)', weight='bold', ha='left', va='bottom',transform=axs5.transAxes)
axs5.axes.set_xbound(-2.1, 2.1)
axs5.axes.set_ybound(-2.5, 2.5)

# forecasting phase V_1
axs6.set_title('forecasting phase')
axs6.set_xticks(xlabel)
axs6.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,doubleScroll.data.y[0,doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs6.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,productRepresentation.prediction[0,0:plottime_pts],linewidth=t_linewidth,color='r')
axs6.axes.xaxis.set_ticklabels([])
axs6.axes.set_xbound(97,303)
axs6.axes.set_ybound(-2.1, 2.1)

# forecasting phase V_2
axs7.set_xticks(xlabel)
axs7.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,doubleScroll.data.y[1,doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs7.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,productRepresentation.prediction[1,0:plottime_pts],linewidth=t_linewidth,color='r')
axs7.axes.xaxis.set_ticklabels([])
axs7.axes.set_xbound(97,303)
axs7.axes.set_ybound(-1.1, 1.1)

# forecasting phase I
axs8.set_xticks(xlabel)
axs8.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,doubleScroll.data.y[2,doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs8.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,productRepresentation.prediction[2,0:plottime_pts],linewidth=t_linewidth,color='r')
axs8.set_xlabel('Time')
axs8.axes.set_xbound(97,303)
axs8.axes.set_ybound(-2.5, 2.5)

# predicted attractor with Distributed Representration
axs9.plot(distributedRepresentation.prediction[0,:],distributedRepresentation.prediction[2,:],linewidth=a_linewidth,color='g')
axs9.set_xlabel('$V_1$')
axs9.set_ylabel('$I$')
axs9.set_title('Distributed; ' + r'$D=34$')
axs9.text(-.35,0.92,'f)', weight='bold', ha='left', va='bottom',transform=axs9.transAxes)
axs9.axes.set_xbound(-2.1, 2.1)
axs9.axes.set_ybound(-2.5, 2.5)

# forecasting phase V_1
axs10.set_title('forecasting phase')
axs10.set_xticks(xlabel)
axs10.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,doubleScroll.data.y[0,doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs10.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,distributedRepresentation.prediction[0,0:plottime_pts],linewidth=t_linewidth,color='g')
axs10.axes.xaxis.set_ticklabels([])
axs10.axes.set_xbound(97,303)
axs10.axes.set_ybound(-2.1, 2.1)

# forecasting phase V_2
axs11.set_xticks(xlabel)
axs11.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,doubleScroll.data.y[1,doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs11.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,distributedRepresentation.prediction[1,0:plottime_pts],linewidth=t_linewidth,color='g')
axs11.axes.xaxis.set_ticklabels([])
axs11.axes.set_xbound(97,303)
axs11.axes.set_ybound(-1.1, 1.1)

# forecasting phase I
axs12.set_xticks(xlabel)
axs12.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,doubleScroll.data.y[2,doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1],linewidth=t_linewidth, color='k')
axs12.plot(doubleScroll.t_eval[doubleScroll.warmtrain_pts-1:doubleScroll.warmtrain_pts+plottime_pts-1]-doubleScroll.warmup,distributedRepresentation.prediction[2,0:plottime_pts],linewidth=t_linewidth,color='g')
axs12.set_xlabel('Time')
axs12.axes.set_xbound(97,303)
axs12.axes.set_ybound(-2.5, 2.5)

fig1.align_ylabels([axs2,axs3,axs4])
plt.savefig('./results/Figure_3_d_f_DoubleScroll.png')