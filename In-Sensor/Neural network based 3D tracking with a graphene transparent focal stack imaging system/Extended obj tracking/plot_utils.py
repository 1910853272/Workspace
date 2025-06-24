import matplotlib.pyplot as plt
def show2D(trueXYZobj_bat,predXYZobj_bat,howmany=None,save_name=None):   
    """
    code for plotting the 3d points coordinates predicted by the network in 2D perspective
    Input:
    trueXYZobj_bat：torch tensor with dimension N,3
    predXYZobj_bat: torch tensor with dimension N,3
    subset:wether to plot subset of the provided data points, e.g, None or a number indicating how many points to be plotted 
    """
    fig1=plt.figure()
    plt.scatter(trueXYZobj_bat.numpy()[:howmany,0],trueXYZobj_bat.numpy()[:howmany,1],marker='o',label='true')
    plt.scatter(predXYZobj_bat.numpy()[:howmany,0],predXYZobj_bat.numpy()[:howmany,1],marker='o',label='pred')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('X-Y')
    if save_name!= None:
        plt.savefig(save_name+'(X-Y)')
    
    plt.show()


    fig2=plt.figure()
    plt.scatter(trueXYZobj_bat.numpy()[:howmany,0],trueXYZobj_bat.numpy()[:howmany,2],marker='o',label='true')
    plt.scatter(predXYZobj_bat.numpy()[:howmany,0],predXYZobj_bat.numpy()[:howmany,2],marker='o',label='pred')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('X-Z')
    if save_name!= None:
        plt.savefig(save_name+'(X-Z)')
    plt.show()
    

    fig3=plt.figure()
    plt.scatter(trueXYZobj_bat.numpy()[:howmany,1],trueXYZobj_bat.numpy()[:howmany,2],marker='o',label='true')
    plt.scatter(predXYZobj_bat.numpy()[:howmany,1],predXYZobj_bat.numpy()[:howmany,2],marker='o',label='pred')
    plt.legend()
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('Y-Z')
    if save_name!= None:
        plt.savefig(save_name+'(Y-Z)')
    plt.show()