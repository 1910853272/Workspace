# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in Figure S.11.

"""

# Initial version of this script comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code

import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import Lorenz63Noisy
from ProductRepresentation import ProductRepresentationRC
from DistributedRepresentation import DistributedRepresentationRC
from scipy.integrate import solve_ivp

##
## Data - Lorenz63 system
##
lorenz_noisy = Lorenz63Noisy()
print('standard deviation for x,y,z: '+str(np.std(lorenz_noisy.data.y[0,:]))+' '+str(np.std(lorenz_noisy.data.y[1,:]))+' '+str(np.std(lorenz_noisy.data.y[2,:])))

# Noiseless solution for testing
def lorenz_nn(t, y):
    return [lorenz_noisy.sigma * (y[1] - y[0]),  y[0] * (lorenz_noisy.rho - y[2]) - y[1] , y[0] * y[1] - lorenz_noisy.betaLor * y[2] ]
lorenz_soln_nn = solve_ivp(lorenz_nn, (0,lorenz_noisy.testtime), lorenz_noisy.data.y[:,lorenz_noisy.warmtrain_pts-1] , t_eval=np.linspace(0,lorenz_noisy.testtime,lorenz_noisy.testtime_pts+1), method='RK23')
total_var_nn=np.var(lorenz_soln_nn.y[0,:])+np.var(lorenz_soln_nn.y[1,:])+np.var(lorenz_soln_nn.y[2,:])

##
## RC with product representation
##
# ridge parameter for regression
ridge_param = 1.4e-2

productRepresentation = ProductRepresentationRC(system=lorenz_noisy, k=2, ridge_param=ridge_param)
productRepresentation.fit12()
productRepresentation.predict12()

productRepresentation_test_nrmse = np.sqrt(np.mean((lorenz_soln_nn.y[0:productRepresentation.d,0:lorenz_noisy.lyaptime_pts]-productRepresentation.prediction[0:productRepresentation.d,0:lorenz_noisy.lyaptime_pts])**2)/total_var_nn)
print('Product representation, test NRMSE: '+str(productRepresentation_test_nrmse))

##
## RC with distributed representation 
##        
# ridge parameter for regression
ridge_param_dr = 1.e-5

distributedRepresentation = DistributedRepresentationRC(system=lorenz_noisy, D=28, k=2, ridge_param=ridge_param_dr, seed = 1)
distributedRepresentation.fit12()
distributedRepresentation.predict12()

distributedRepresentation_test_nrmse = np.sqrt(np.mean((lorenz_soln_nn.y[0:distributedRepresentation.d,0:lorenz_noisy.lyaptime_pts]-distributedRepresentation.prediction[0:distributedRepresentation.d,0:lorenz_noisy.lyaptime_pts])**2)/total_var_nn)
print('Distributed representation, test NRMSE: '+str(distributedRepresentation_test_nrmse))


##
## Plot
##
# how much of testtime to plot
plottime=20.
plottime_pts=round(plottime/lorenz_noisy.dt)

t_linewidth=1.1
a_linewidth=0.3
plt.rcParams.update({'font.size': 12})

fig1 = plt.figure(dpi=300)
fig1.set_figheight(8)
fig1.set_figwidth(12.9)

xlabel=[10,15,20,25,30,35,40]
h=140
w=150

# top left of grid is 0,0
axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 9), colspan=22, rowspan=36) 
axs2 = plt.subplot2grid(shape=(h,w), loc=(52, 0), colspan=42, rowspan=20)
axs3 = plt.subplot2grid(shape=(h,w), loc=(75, 0), colspan=42, rowspan=20)
axs4 = plt.subplot2grid(shape=(h,w), loc=(98, 0), colspan=42, rowspan=20)

# Product Representration predictions
axs5 = plt.subplot2grid(shape=(h,w), loc=(0, 61), colspan=22, rowspan=36)
axs6 = plt.subplot2grid(shape=(h,w), loc=(52, 50),colspan=42, rowspan=20)
axs7 = plt.subplot2grid(shape=(h,w), loc=(75, 50), colspan=42, rowspan=20)
axs8 = plt.subplot2grid(shape=(h,w), loc=(98, 50), colspan=42, rowspan=20)

# Distributed Representration predictions
axs9 = plt.subplot2grid(shape=(h,w), loc=(0, 111), colspan=22, rowspan=36)
axs10 = plt.subplot2grid(shape=(h,w), loc=(52, 100),colspan=42, rowspan=20)
axs11 = plt.subplot2grid(shape=(h,w), loc=(75, 100), colspan=42, rowspan=20)
axs12 = plt.subplot2grid(shape=(h,w), loc=(98, 100), colspan=42, rowspan=20)

# true NOISY Lorenz attractor
axs1.plot(lorenz_noisy.data.y[0,lorenz_noisy.warmtrain_pts:lorenz_noisy.maxtime_pts],lorenz_noisy.data.y[2,lorenz_noisy.warmtrain_pts:lorenz_noisy.maxtime_pts],linewidth=a_linewidth, color='k')
axs1.set_xlabel('x', style='italic')
axs1.set_ylabel('z', style='italic')
axs1.set_title('Noise-driven \n ground truth')
axs1.text(-.25,.92,'a)',weight='bold', ha='left', va='bottom',transform=axs1.transAxes)
axs1.axes.set_xbound(-21,21)
axs1.axes.set_ybound(2,48)

# training phase x
axs2.set_title('Noise-driven training') 
axs2.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,lorenz_noisy.data.y[0,lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs2.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,productRepresentation.prediction_train[0,:],linewidth=t_linewidth, color='r', linestyle='dashed')
axs2.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,distributedRepresentation.prediction_train[0,:],linewidth=t_linewidth, color='g', linestyle='dashdot')
axs2.set_ylabel('x', style='italic')
axs2.text(-.155*1.2,0.87,'b)',weight='bold', ha='left', va='bottom',transform=axs2.transAxes)
axs2.axes.xaxis.set_ticklabels([])
axs2.axes.set_ybound(-21.,21.)
axs2.axes.set_xbound(-.15,10.15)

# training phase y
axs3.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,lorenz_noisy.data.y[1,lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs3.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,productRepresentation.prediction_train[1,:],linewidth=t_linewidth,color='r', linestyle='dashed')
axs3.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,distributedRepresentation.prediction_train[1,:],linewidth=t_linewidth,color='g', linestyle='dashdot')
axs3.set_ylabel('y', style='italic')
axs3.text(-.155*1.2,0.87,'c)',weight='bold', ha='left', va='bottom',transform=axs3.transAxes)
axs3.axes.xaxis.set_ticklabels([])
axs3.axes.set_xbound(-.15,10.15)
axs3.axes.set_ybound(-25,25)

# training phase z
axs4.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,lorenz_noisy.data.y[2,lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts],linewidth=2*t_linewidth, linestyle='dotted', color='k')
axs4.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,productRepresentation.prediction_train[2,:],linewidth=t_linewidth,color='r', linestyle='dashed')
axs4.plot(lorenz_noisy.t_eval[lorenz_noisy.warmup_pts:lorenz_noisy.warmtrain_pts]-lorenz_noisy.warmup,distributedRepresentation.prediction_train[2,:],linewidth=t_linewidth,color='g', linestyle='dashdot')
axs4.set_ylabel('z', style='italic')
axs4.text(-.155*1.2,0.87,'d)',weight='bold', ha='left', va='bottom',transform=axs4.transAxes)
axs4.set_xlabel('Time')
axs4.axes.set_xbound(-.15,10.15)

# predicted attractor with Product Representration
axs5.plot(productRepresentation.prediction[0,:],productRepresentation.prediction[2,:],linewidth=a_linewidth,color='r')
axs5.set_xlabel('x', style='italic')
axs5.set_ylabel('z', style='italic')
axs5.set_title('Product; ' + r'$D=28$' +'\n')
axs5.text(-.25,0.92,'e)',weight='bold', ha='left', va='bottom',transform=axs5.transAxes)
axs5.axes.set_xbound(-21,21)
axs5.axes.set_ybound(2,48)

# testing phase x
axs6.set_title('Noise-free forecasting')
axs6.set_xticks(xlabel)
axs6.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,lorenz_soln_nn.y[0,0:plottime_pts],linewidth=t_linewidth, color='k')
axs6.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,productRepresentation.prediction[0,0:plottime_pts],linewidth=t_linewidth,color='r', linestyle='dashed')
axs6.text(-.155*1.15,0.87,'f)',weight='bold', ha='left', va='bottom',transform=axs6.transAxes)
axs6.axes.xaxis.set_ticklabels([])
axs6.axes.set_xbound(9.7,30.3)

# testing phase y
axs7.set_xticks(xlabel)
axs7.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,lorenz_soln_nn.y[1,0:plottime_pts],linewidth=t_linewidth, color='k')
axs7.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,productRepresentation.prediction[1,0:plottime_pts],linewidth=t_linewidth,color='r', linestyle='dashed')
axs7.text(-.155*1.15,0.87,'g)',weight='bold', ha='left', va='bottom',transform=axs7.transAxes)
axs7.axes.xaxis.set_ticklabels([])
axs7.axes.set_xbound(9.7,30.3)
axs7.axes.set_ybound(-25,25)

# testing phase z
axs8.set_xticks(xlabel)
axs8.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,lorenz_soln_nn.y[2,0:plottime_pts],linewidth=t_linewidth, color='k')
axs8.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,productRepresentation.prediction[2,0:plottime_pts],linewidth=t_linewidth,color='r', linestyle='dashed')
axs8.text(-.155*1.15,0.87,'h)',weight='bold', ha='left', va='bottom',transform=axs8.transAxes)
axs8.set_xlabel('Time')
axs8.axes.set_xbound(9.7,30.3)

# predicted attractor with Distributed Representration
axs9.plot(distributedRepresentation.prediction[0,:],distributedRepresentation.prediction[2,:],linewidth=a_linewidth,color='g')
axs9.set_xlabel('x', style='italic')
axs9.set_ylabel('z', style='italic')
axs9.set_title('Distributed; ' + r'$D=28$' +'\n')
axs9.text(-.25,0.92,'i)',weight='bold', ha='left', va='bottom',transform=axs9.transAxes)
axs9.axes.set_xbound(-21,21)
axs9.axes.set_ybound(2,48)

# testing phase x
axs10.set_title('Noise-free forecasting')
axs10.set_xticks(xlabel)
axs10.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,lorenz_soln_nn.y[0,0:plottime_pts],linewidth=t_linewidth, color='k')
axs10.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,distributedRepresentation.prediction[0,0:plottime_pts],linewidth=t_linewidth,color='g', linestyle='dashed')
axs10.text(-.155*1.15,0.87,'j)',weight='bold', ha='left', va='bottom',transform=axs10.transAxes)
axs10.axes.xaxis.set_ticklabels([])
axs10.axes.set_xbound(9.7,30.3)
axs10.axes.set_ybound(-21.,21.)

# testing phase y
axs11.set_xticks(xlabel)
axs11.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,lorenz_soln_nn.y[1,0:plottime_pts],linewidth=t_linewidth, color='k')
axs11.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,distributedRepresentation.prediction[1,0:plottime_pts],linewidth=t_linewidth,color='g', linestyle='dashed')
axs11.text(-.155*1.15,0.87,'k)',weight='bold', ha='left', va='bottom',transform=axs11.transAxes)
axs11.axes.xaxis.set_ticklabels([])
axs11.axes.set_xbound(9.7,30.3)
axs11.axes.set_ybound(-25,25)

# testing phase z
axs12.set_xticks(xlabel)
axs12.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,lorenz_soln_nn.y[2,0:plottime_pts],linewidth=t_linewidth, color='k')
axs12.plot(lorenz_noisy.t_eval[lorenz_noisy.warmtrain_pts-1:lorenz_noisy.warmtrain_pts+plottime_pts-1]-lorenz_noisy.warmup,distributedRepresentation.prediction[2,0:plottime_pts],linewidth=t_linewidth,color='g', linestyle='dashed')
axs12.text(-.155*1.15,0.87,'l)',weight='bold', ha='left', va='bottom',transform=axs12.transAxes)
axs12.set_xlabel('Time')
axs12.axes.set_xbound(9.7,30.3)

fig1.align_ylabels([axs2,axs3,axs4])
plt.savefig('/results/Figure_S11_Lorenz_noise.png')
plt.show()