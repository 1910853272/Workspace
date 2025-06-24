# -*- coding: utf-8 -*-
"""

This script performs the experiment reported in panels a, b, & c of Figure 3.

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import KuramotoSivashinskyLong
from DistributedRepresentation import DistributedRepresentationRCTorch
from ProductRepresentation import ProductRepresentationRCTorch
from EchoStateNetwork import EchoStateNetworkRCTorch

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def colorbar_invis(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    cbar.remove() 

device = torch.device("cpu")

##
## Data - Kuramoto-Sivashinsky system
##
kuramoto = KuramotoSivashinskyLong(device=device)
kuramoto.assign_data(seed = 20)
dataKS = kuramoto.data.y.detach().cpu().numpy()

##
## RC with product representation
##
productRepresentation = ProductRepresentationRCTorch(system=kuramoto, device = device, k=2, ridge_param=1.e-4)
productRepresentation.reservoir123()
productRepresentation.readout()   
productRepresentation.nrmse_train = productRepresentation.nrmse()  
productRepresentation.predict123()
print('Product representation, training NRMSE: '+str(productRepresentation.nrmse_train))
print('Product representation, test NRMSE: '+str(productRepresentation.nrmse_test))


##
## RC with distributed representation 
##
distributedRepresentation = DistributedRepresentationRCTorch(system=kuramoto, device = device, D=4000, k=2, normalize = False, ridge_param=1.e-6, seed = 14) 
distributedRepresentation.reservoir123()
distributedRepresentation.readout()   
distributedRepresentation.nrmse_train = distributedRepresentation.nrmse()  
distributedRepresentation.predict123()    
print('Distributed representation, training NRMSE: '+str(distributedRepresentation.nrmse_train))
print('Distributed representation, test NRMSE: '+str(distributedRepresentation.nrmse_test))

##
## RC with echo state networks 
##
#for i in range(0,30):  
echoStateNetwork = EchoStateNetworkRCTorch(system=kuramoto, device = device, D = 4000, input_scale = 0.1, spectral_radius = 0.1, include_bias = True, normalize = True, ridge_param=1.e-8,  seed_id = 15)    
echoStateNetwork.reservoir_train()    
echoStateNetwork.readout()   
echoStateNetwork.nrmse_train = echoStateNetwork.nrmse()  
echoStateNetwork.predict()
#    print('Seed: ' + str(i) + '; Echo state network, test NRMSE: {price: .3f}'.format(price = echoStateNetwork.nrmse_test))  
print('Echo state network, training NRMSE: '+str(echoStateNetwork.nrmse_train))
print('Echo state network, test NRMSE: '+str(echoStateNetwork.nrmse_test))

predPR = productRepresentation.prediction.detach().cpu().numpy()
predDR = distributedRepresentation.prediction.detach().cpu().numpy()
predESN = echoStateNetwork.prediction.detach().cpu().numpy()

##
## Plot
##
plt.rcParams.update({'font.size': 10})
pts_to_plot=400
fig1 = plt.figure(figsize=(8.,4), dpi=300)

plt.subplot(3, 3, 1)
plt.imshow(dataKS[:,kuramoto.warmtrain_pts:kuramoto.warmtrain_pts+pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
plt.xticks([])
plt.yticks(ticks=np.arange(0,16,4) , labels=np.arange(1,17,4))
plt.title('Echo State Net; ' + r'$D=4,000$')
plt.ylabel('Ground truth')
plt.text(-87,-4.,'a)', weight='bold', ha='left', va='top')

plt.subplot(3, 3, 4)
plt.imshow(predESN[0:echoStateNetwork.d,0:pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
plt.xticks([])
plt.yticks(ticks=np.arange(0,16,4) , labels=np.arange(1,17,4))
plt.ylabel('Prediction')

plt.subplot(3, 3, 7)
plt.imshow(dataKS[:,kuramoto.warmtrain_pts:kuramoto.warmtrain_pts+pts_to_plot]-predESN[0:echoStateNetwork.d,0:pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
plt.xticks(ticks=np.round(np.arange(0.,11.)*kuramoto.lyaptime/kuramoto.dt).astype('int') , labels=np.arange(0,11))
plt.yticks(ticks=np.arange(0,16,4) , labels=np.arange(1,17,4))
plt.xlabel('Lyapunov time')
plt.ylabel('Error')

plt.subplot(3, 3, 2)
plt.imshow(dataKS[:,kuramoto.warmtrain_pts:kuramoto.warmtrain_pts+pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
plt.yticks([])
plt.xticks([])
plt.title('Product; ' + r'$D=6,545$')
plt.text(-35,-4.,'b)', weight='bold', ha='left', va='top')

plt.subplot(3, 3, 5)
plt.imshow(predPR[0:productRepresentation.d,0:pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 8)
plt.imshow(dataKS[:,kuramoto.warmtrain_pts:kuramoto.warmtrain_pts+pts_to_plot]-predPR[0:productRepresentation.d,0:pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
plt.xticks(ticks=np.round(np.arange(0.,11.)*kuramoto.lyaptime/kuramoto.dt).astype('int') , labels=np.arange(0,11))
plt.yticks([])
plt.xlabel('Lyapunov time')

plt.subplot(3, 3, 3)
img1 = plt.imshow(dataKS[:,kuramoto.warmtrain_pts:kuramoto.warmtrain_pts+pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
colorbar_invis(img1)
plt.xticks([])
plt.yticks([])
plt.title('Distributed; ' + r'$D=4,000$')
plt.text(-35,-4.,'c)', weight='bold', ha='left', va='top')

plt.subplot(3, 3, 6)
img1 = plt.imshow(predDR[0:distributedRepresentation.d,0:pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
colorbar(img1)
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 9)
img1 = plt.imshow(dataKS[:,kuramoto.warmtrain_pts:kuramoto.warmtrain_pts+pts_to_plot]-predDR[0:distributedRepresentation.d,0:pts_to_plot], aspect='auto', cmap='jet')
plt.clim(vmin=-3, vmax=3)
colorbar_invis(img1)
plt.xticks(ticks=np.round(np.arange(0.,11.)*kuramoto.lyaptime/kuramoto.dt).astype('int') , labels=np.arange(0,11))
plt.yticks([])
plt.xlabel('Lyapunov time')

plt.tight_layout()
plt.savefig('/results/Figure_S1_Kuramoto_Sivashinsky.png',bbox_inches="tight")
plt.show()