import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as dset
from scipy.io import loadmat
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from operator import add

class depthnet(nn.Module):
    def __init__(self):
        super(depthnet, self).__init__()
        self.FC_front= nn.Linear(16, 1,bias=True)
        self.FC_back= nn.Linear(16, 1,bias=True)
        self.end_regress=nn.Linear(2,1,bias=True)
            
    def forward(self,x):
        N,nF,H,W=x.shape
        front_feature=self.FC_front(x[:,0,:,:].contiguous().view(N,-1))#output is N,1
        back_feature=self.FC_back(x[:,1,:,:].contiguous().view(N,-1)) #output is N,1
        Combined_feature=torch.cat((front_feature, back_feature), 1) #output is N,2, with first column is the feature from front plane
        out=self.end_regress(Combined_feature)#out is N,1       
        return out
    
class depthnet2(nn.Module):
    #This net should be funcitonally equalvalent to depthnet1
    def __init__(self):
        super(depthnet2, self).__init__(input_dim=32)
        self.Linear1=nn.Linear(input_dim,1,bias=True)
    def forward(self,x):
        N,nF,H,W=x.shape
        return self.Linear1(x.view(N,-1))
    

class depthnet3(nn.Module):
    ##with more network power and complexity, seems to overfitting the dataset and the test loss is high
    def __init__(self,input_dim=32,num_features=16):
        super(depthnet3, self).__init__()
        self.Linear1=nn.Linear(input_dim,num_features,bias=True)
        self.Linear2=nn.Linear(num_features,1,bias=True)
        
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=self.Linear2(x)

        return x

    
class depthnet4(nn.Module):
    ##with more network power and complexity, seems to overfitting the dataset and the test loss is high
    def __init__(self,input_dim=32,num_features=16):
        super(depthnet4, self).__init__()
        self.Linear1=nn.Linear(input_dim,num_features,bias=True)
        self.Linear2=nn.Linear(num_features,num_features,bias=True)
        self.Linear3=nn.Linear(num_features,1,bias=True)
      
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=F.relu(self.Linear2(x))
        x=self.Linear3(x)

        return x    
    
class depthnet4_xyz(nn.Module):
    ##predicting x,y,z using single net
    def __init__(self,input_dim=32,num_features=16):
        super(depthnet4_xyz, self).__init__()
        self.Linear1=nn.Linear(input_dim, num_features,bias=True)
        self.Linear2=nn.Linear(num_features,num_features,bias=True)
        self.Linear3=nn.Linear(num_features,3,bias=True)
      
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=F.relu(self.Linear2(x))
        x=self.Linear3(x)

        return x    
    
class depthnet5_xyz(nn.Module):
    ##predicting x,y,z using single net
    def __init__(self,input_dim=32,num_features=16):
        super(depthnet5_xyz, self).__init__()
        self.Linear1=nn.Linear(input_dim, num_features,bias=True)
        self.Linear2=nn.Linear(num_features,num_features,bias=True)
        self.Linear3=nn.Linear(num_features,num_features,bias=True)
        self.Linear4=nn.Linear(num_features,3,bias=True)
           
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=F.relu(self.Linear2(x))
        x=F.relu(self.Linear3(x))
        x=self.Linear4(x)

        return x    

    
    
class FSdataset(dset):
    ##the data should has two matrix, FS and xyz. FS has dimension N,nF,x,y, xyz has dimension N,3. Where 3 columns are coordinates of object x,y,z respectively. 
    """
    data_idx:specify what subset of the fulldata is used to split into train/test, 
    e.g, for a particular fixed xy and sweep z,e.g, np.arange(0,1999,100)
    for a particular fixed z and sweep xy, np.arange(100,200,1)
    test_idx specify among the subset of the fulldata, which point is used as the testset (start from 1, ie, test_idx=1 means FS_all[0] )
    
    return a dictionary containing FS (N,nF,nx,ny) and truexyzobj_all (N,3)
    """
    def __init__(self,folderpath_img,set,data_idx=1,test_idx=1,scale_z=False):
        super(FSdataset,self).__init__() 
        if set=='Train':
            self.truexyzobj_all=loadmat(folderpath_img)["xyz"]
            mask = np.zeros(len(self.truexyzobj_all),dtype=bool)
            mask[data_idx]=1
            mask[data_idx[test_idx-1]]=False
            self.FS_all=loadmat(folderpath_img)["FS"][mask] #8,2,4,4 
            self.truexyzobj_all=self.truexyzobj_all[mask]

        elif set=='Test':
            self.FS_all=loadmat(folderpath_img)["FS"][[data_idx[test_idx-1]]]
            self.truexyzobj_all=loadmat(folderpath_img)["xyz"][[data_idx[test_idx-1]]]

        elif set=='Full':
            self.FS_all=loadmat(folderpath_img)["FS"]#8,2,4,4 
            self.truexyzobj_all=loadmat(folderpath_img)["xyz"]
            
        if scale_z == True:
            self.truexyzobj_all[:,2]/=33.3 
            
        
        
    def __len__(self):
        return len(self.truexyzobj_all) #number of samples
    def __getitem__(self,index):
        
        FS= self.FS_all[index]  ##each sample has shape 2,4,4
        truexyzobj_all=self.truexyzobj_all[index] ##each sample has shape 1
             
        return {'FS':FS,'xyz':truexyzobj_all}



    
    
    


def showFS(dataset):
    """Show the focal stack data of the dataset, first column is the front focal plane,"second column is the back focal plane"""
    ##usage example: showFS(train_dataset)
    fig = plt.figure(1, (len(dataset), len(dataset)))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(len(dataset), 2),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for i in range(len(dataset)):
        for j in range(2):
            grid[i*2+j].imshow(dataset[i]['FS'][j])  # The AxesGrid object work as a list of axes.

    plt.show()