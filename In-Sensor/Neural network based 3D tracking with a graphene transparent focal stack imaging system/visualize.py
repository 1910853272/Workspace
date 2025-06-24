#%matplotlib widget
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
def show3Dpred(trueXYZobj_bat,predXYZobj_bat,howmany=None,save_name=None):
    """
    code for plotting the 3d points coordinates predicted by the network
    Input:
    trueXYZobj_bat：torch tensor with dimension N,3
    predXYZobj_bat: torch tensor with dimension N,3
    subset:wether to plot subset of the provided data points, e.g, None or a number indicating how many points to be plotted 
    """
    if howmany==None:
        howmany=len(trueXYZobj_bat)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X,Y,Z={},{},{}
    X['true'],Y['true'],Z['true']=trueXYZobj_bat.numpy()[:howmany,0],trueXYZobj_bat.numpy()[:howmany,1],trueXYZobj_bat.numpy()[:howmany,2]
    X['pred'],Y['pred'],Z['pred']=predXYZobj_bat.numpy()[:howmany,0],predXYZobj_bat.numpy()[:howmany,1],predXYZobj_bat.numpy()[:howmany,2]
    for c, m, group in [('r', 'o','true'), ('b', '^', 'pred')]:
        xs = X[group]
        ys = Y[group]
        zs = Z[group]
        ax.scatter(xs, ys, zs, c=c, marker=m,label=group)

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    plt.legend()
    if save_name!= None:
        plt.savefig(save_name)
        
    plt.show()
    
def show3Dpred_2objs(trueXYZobj_bat,predXYZobj_bat,howmany=None,save_name=None):
    """
    code for plotting the 3d points coordinates predicted by the network
    Input:
    trueXYZobj_bat：torch tensor with dimension N,6
    predXYZobj_bat: torch tensor with dimension N,6
    subset:wether to plot subset of the provided data points, e.g, None or a number indicating how many points to be plotted 
    """
    if howmany==None:
        howmany=len(trueXYZobj_bat)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X1,Y1,Z1={},{},{}
    X2,Y2,Z2={},{},{}
    X1['true'],Y1['true'],Z1['true']=trueXYZobj_bat.numpy()[:howmany,0],trueXYZobj_bat.numpy()[:howmany,1],trueXYZobj_bat.numpy()[:howmany,2]
    X1['pred'],Y1['pred'],Z1['pred']=predXYZobj_bat.numpy()[:howmany,0],predXYZobj_bat.numpy()[:howmany,1],predXYZobj_bat.numpy()[:howmany,2]
    X2['true'],Y2['true'],Z2['true']=trueXYZobj_bat.numpy()[:howmany,3],trueXYZobj_bat.numpy()[:howmany,4],trueXYZobj_bat.numpy()[:howmany,5]
    X2['pred'],Y2['pred'],Z2['pred']=predXYZobj_bat.numpy()[:howmany,3],predXYZobj_bat.numpy()[:howmany,4],predXYZobj_bat.numpy()[:howmany,5]   
    
    for c, m, group in [('r', 'o','true'), ('b', 'o', 'pred')]:
        x1s = X1[group]
        y1s = Y1[group]
        z1s = Z1[group]
        x2s = X2[group]
        y2s = Y2[group]
        z2s = Z2[group]        
        for i in range(howmany):
            #ax.scatter(x1s[i], ys[i], zs[i], c=c, marker=m,label=group)
            #ax.scatter(x1s[i], ys[i], zs[i], c=c, marker=m,label=group)
            ax.plot([x1s[i],x2s[i]], [y1s[i],y2s[i]], [z1s[i],z2s[i]],c=c, marker = m,label = group)
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    
    #Trick to removing repeated labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    if save_name!= None:
        plt.savefig(save_name)
        
    plt.show()
    

    
    
    
def show2D(trueXYZobj_bat,predXYZobj_bat,howmany=None,save_name=None,offset=0):   
    """
    code for plotting the 3d points coordinates predicted by the network in 2D perspective
    Input:
    trueXYZobj_bat：torch tensor with dimension N,3
    predXYZobj_bat: torch tensor with dimension N,3
    subset:wether to plot subset of the provided data points, e.g, None or a number indicating how many points to be plotted 
    """
    fig1=plt.figure()
    plt.scatter(trueXYZobj_bat.numpy()[offset:howmany+offset,0],trueXYZobj_bat.numpy()[offset:howmany+offset,1],marker='o')#,label='true')
    plt.scatter(predXYZobj_bat.numpy()[offset:howmany+offset,0],predXYZobj_bat.numpy()[offset:howmany+offset,1],marker='o')#,label='pred')
    #plt.legend()
    plt.xlabel('X (mm)',fontsize = 18)
    plt.ylabel('Y (mm)',fontsize = 18)
    plt.title('X-Y')
    if save_name!= None:
        plt.savefig(save_name+'(X-Y)')
    
    plt.show()


    fig2=plt.figure()
    plt.scatter(trueXYZobj_bat.numpy()[offset:howmany+offset,0],trueXYZobj_bat.numpy()[offset:howmany+offset,2],marker='o')#,label='true')
    plt.scatter(predXYZobj_bat.numpy()[offset:howmany+offset,0],predXYZobj_bat.numpy()[offset:howmany+offset,2],marker='o')#,label='pred')
    #plt.legend()
    plt.xlabel('X (mm)',fontsize = 18)
    plt.ylabel('Z (mm)',fontsize = 18)
    plt.title('X-Z')
    if save_name!= None:
        plt.savefig(save_name+'(X-Z)')
    plt.show()
    

    fig3=plt.figure()
    plt.scatter(trueXYZobj_bat.numpy()[offset:howmany+offset,1],trueXYZobj_bat.numpy()[offset:howmany+offset,2],marker='o')#,label='true')
    plt.scatter(predXYZobj_bat.numpy()[offset:howmany+offset,1],predXYZobj_bat.numpy()[offset:howmany+offset,2],marker='o')#,label='pred')
    #plt.legend()
    plt.xlabel('Y (mm)',fontsize = 18)
    plt.ylabel('Z (mm)',fontsize = 18)
    plt.title('Y-Z')
    if save_name!= None:
        plt.savefig(save_name+'(Y-Z)')
    plt.show()
    
def show2D_2objs(trueXYZobj_bat,predXYZobj_bat,howmany=None,save_name=None,offset=0):   
    """
    code for plotting the 3d points coordinates predicted by the network in 2D perspective
    Input:
    trueXYZobj_bat：torch tensor with dimension N,6 (x1,y1,z1,x2,y2,z2)
    predXYZobj_bat: torch tensor with dimension N,6 (x1,y1,z1,x2,y2,z2)
    subset:wether to plot subset of the provided data points, e.g, None or a number indicating how many points to be plotted 
    """
    fig1=plt.figure()
    for i in range(offset, howmany+offset):
        plt.plot([trueXYZobj_bat[i,0],trueXYZobj_bat[i,3]], [trueXYZobj_bat[i,1],trueXYZobj_bat[i,4]], 'o-',c='C0',label='true')
        plt.plot([predXYZobj_bat[i,0],predXYZobj_bat[i,3]], [predXYZobj_bat[i,1],predXYZobj_bat[i,4]], 'o-',c='C1',label='pred')
    #plt.legend(['true','pred'])
    plt.xlabel('X (mm)',fontsize = 18)
    plt.ylabel('Y (mm)',fontsize = 18)
    plt.title('X-Y')
    if save_name!= None:
        plt.savefig(save_name+'(X-Y)')
    
        plt.show()
        
    fig2=plt.figure()
    for i in range(offset, howmany+offset):
        plt.plot([trueXYZobj_bat[i,0],trueXYZobj_bat[i,3]], [trueXYZobj_bat[i,2],trueXYZobj_bat[i,5]], 'o-',c='C0',label='true')
        plt.plot([predXYZobj_bat[i,0],predXYZobj_bat[i,3]], [predXYZobj_bat[i,2],predXYZobj_bat[i,5]], 'o-',c='C1',label='pred')
    #plt.legend(['true','pred'])
    plt.xlabel('X (mm)',fontsize = 18)
    plt.ylabel('Z (mm)',fontsize = 18)
    plt.title('X-Z (mm)')
    if save_name!= None:
        plt.savefig(save_name+'(X-Z)')
    
        plt.show()
        
    fig3=plt.figure()
    for i in range(offset, howmany+offset):
        plt.plot([trueXYZobj_bat[i,1],trueXYZobj_bat[i,4]], [trueXYZobj_bat[i,2],trueXYZobj_bat[i,5]], 'o-',c='C0',label='true')
        plt.plot([predXYZobj_bat[i,1],predXYZobj_bat[i,4]], [predXYZobj_bat[i,2],predXYZobj_bat[i,5]], 'o-',c='C1',label='pred')
    #plt.legend(['true','pred'])
    plt.xlabel('Y (mm)',fontsize = 18)
    plt.ylabel('Z (mm)',fontsize = 18)
    plt.title('Y-Z')
    if save_name!= None:
        plt.savefig(save_name+'(Y-Z)')
    
        plt.show()
def show2D_3objs(trueXYZobj_bat,predXYZobj_bat,howmany=None,save_name=None,offset=0):   
    """
    code for plotting the 3d points coordinates predicted by the network in 2D perspective
    Input:
    trueXYZobj_bat：torch tensor with dimension N,6 (x1,y1,z1,x2,y2,z2,x3,y3,z3)
    predXYZobj_bat: torch tensor with dimension N,6 (x1,y1,z1,x2,y2,z2,x3,y3,z3)
    subset:wether to plot subset of the provided data points, e.g, None or a number indicating how many points to be plotted 
    """
    fig1=plt.figure()
    for i in range(offset, howmany+offset):
        plt.plot([trueXYZobj_bat[i,0],trueXYZobj_bat[i,3],trueXYZobj_bat[i,6],trueXYZobj_bat[i,0]], [trueXYZobj_bat[i,1],trueXYZobj_bat[i,4],trueXYZobj_bat[i,7],trueXYZobj_bat[i,1]], 'o-',c='C0',label='true')
        plt.plot([predXYZobj_bat[i,0],predXYZobj_bat[i,3],predXYZobj_bat[i,6],predXYZobj_bat[i,0]], [predXYZobj_bat[i,1],predXYZobj_bat[i,4],predXYZobj_bat[i,7],predXYZobj_bat[i,1]], 'o-',c='C1',label='pred')
        
    #plt.legend(['true','pred'])
    plt.xlabel('X (mm)',fontsize = 18)
    plt.ylabel('Y (mm)',fontsize = 18)
    plt.title('X-Y')
    if save_name!= None:
        plt.savefig(save_name+'(X-Y)')
    
        plt.show()
        
    fig2=plt.figure()
    for i in range(offset, howmany+offset):
        plt.plot([trueXYZobj_bat[i,0],trueXYZobj_bat[i,3],trueXYZobj_bat[i,6],trueXYZobj_bat[i,0]], [trueXYZobj_bat[i,2],trueXYZobj_bat[i,5],trueXYZobj_bat[i,8],trueXYZobj_bat[i,2]], 'o-',c='C0',label='true')
        plt.plot([predXYZobj_bat[i,0],predXYZobj_bat[i,3],predXYZobj_bat[i,6],predXYZobj_bat[i,0]], [predXYZobj_bat[i,2],predXYZobj_bat[i,5],predXYZobj_bat[i,8],predXYZobj_bat[i,2]], 'o-',c='C1',label='pred')
    #plt.legend(['true','pred'])
    plt.xlabel('X (mm)',fontsize = 18)
    plt.ylabel('Z (mm)',fontsize = 18)
    plt.title('X-Z')
    if save_name!= None:
        plt.savefig(save_name+'(X-Z)')
    
        plt.show()
        
    fig3=plt.figure()
    for i in range(offset, howmany+offset):
        plt.plot([trueXYZobj_bat[i,1],trueXYZobj_bat[i,4],trueXYZobj_bat[i,7],trueXYZobj_bat[i,1]], [trueXYZobj_bat[i,2],trueXYZobj_bat[i,5],trueXYZobj_bat[i,8],trueXYZobj_bat[i,2]], 'o-',c='C0',label='true')
        plt.plot([predXYZobj_bat[i,1],predXYZobj_bat[i,4],predXYZobj_bat[i,7],predXYZobj_bat[i,1]], [predXYZobj_bat[i,2],predXYZobj_bat[i,5],predXYZobj_bat[i,8],predXYZobj_bat[i,2]], 'o-',c='C1',label='pred')
    #plt.legend(['true','pred'])
    plt.xlabel('Y (mm)',fontsize = 18)
    plt.ylabel('Z (mm)',fontsize = 18)
    plt.yticks([-10,-5,0,5,10])
    plt.title('Y-Z')
    if save_name!= None:
        plt.savefig(save_name+'(Y-Z)')
    
        plt.show()

def Y_sweep_trace(dataset,Z_idx=1,X_idx=1,nX=6,nY=6,nZ=11,H=4,W=4):
    """input:
    dataset:(N,nF,H,W) first column is the front focal plane,"second column is the back focal plane， N should vary along X first, then Y, then Z
    Z_idx: a number, indicating among all posible Z values, fix to which one, range 1 to nZ
    X_idx: a number, indicating among all posible X values, fix to which one, range 1 to nX
    nX,nY,nZ: object position numbers along x,y,z
    H,W：Sensor H and W
    assumed to has nF=2
    """
    data_fixed_Z=dataset[np.arange((Z_idx-1)*nX*nY,Z_idx*nX*nY)]['FS'] #index out all FS pairs with a fixed Z, size(nXnY,2,H,W)
    data_fixed_X_Z=data_fixed_Z[np.arange((X_idx-1),nX*nY,nX)] #size nY,2,H,W
    xyz=dataset[np.arange((Z_idx-1)*nX*nY,Z_idx*nX*nY)]['xyz'][np.arange((X_idx-1),nX*nY,nX)]
    
    print(xyz)
    plt.figure()
    plt.plot(xyz[:,1],data_fixed_X_Z.reshape(nY,-1)[:,:H*W])
    plt.title('Y_sweep at X=%.2f,Z=%.2f for Front pixels'%(xyz[0,0],xyz[0,2]))
    plt.xlabel('Y coordinate')
    plt.figure()
    plt.plot(xyz[:,1],data_fixed_X_Z.reshape(nY,-1)[:,H*W:-1])
    plt.title('Y_sweep at X=%.2f,Z=%.2f for Back pixels' %(xyz[0,0],xyz[0,2]))
    plt.xlabel('Y coordinate')
    plt.show()

def X_sweep_trace(dataset,Z_idx=1,Y_idx=1,nX=6,nY=6,nZ=11,H=4,W=4):
    """input:
    dataset:(N,nF,H,W) first column is the front focal plane,"second column is the back focal plane， N should vary along X first, then Y, then Z
    Z_idx: a number, indicating among all posible Z values, fix to which one, range 1 to nZ
    X_idx: a number, indicating among all posible X values, fix to which one, range 1 to nX
    nX,nY,nZ: object position numbers along x,y,z
    H,W：Sensor H and W
    assumed to has nF=2
    """
    data_fixed_Z=dataset[np.arange((Z_idx-1)*nX*nY,Z_idx*nX*nY)]['FS'] #index out all FS pairs with a fixed Z, size(nXnY,2,H,W)
    data_fixed_Y_Z=data_fixed_Z[np.arange((Y_idx-1)*nX,(Y_idx-1)*nX+nX)]  #size nY,2,H,W
    xyz=dataset[np.arange((Z_idx-1)*nX*nY,Z_idx*nX*nY)]['xyz'][np.arange((Y_idx-1)*nX,(Y_idx-1)*nX+nX)] 
    print(xyz)
    plt.figure()
    plt.plot(xyz[:,0],data_fixed_Y_Z.reshape(nY,-1)[:,:H*W])
    plt.title('X_sweep at Y=%.2f,Z=%.2f for Front pixels' %(xyz[0,1],xyz[0,2]))
    plt.xlabel('X coordinate')
    plt.figure()
    plt.plot(xyz[:,0],data_fixed_Y_Z.reshape(nY,-1)[:,H*W:-1])    
    plt.title('X_sweep at Y=%.2f,Z=%.2f for Back pixels' %(xyz[0,1],xyz[0,2]))
    plt.xlabel('X coordinate')
    plt.show()
    
    
    
    
    




