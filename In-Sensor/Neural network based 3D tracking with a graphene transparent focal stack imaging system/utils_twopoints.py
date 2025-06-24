import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as dset
from torch.utils.data import ConcatDataset
from scipy.io import loadmat
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from operator import add
from mpl_toolkits.mplot3d import Axes3D
class FSdataset_twopoints(dset):
    """
    the data should has two matrix, FS and xyz. FS has dimension N,nF,H(y),W(x), xyz has dimension N,3. Where 3 columns are coordinates of object x,y,z respectively. The N should increase along x first, then y, then z. 
   
    return a dictionary containing FS (N,nF,H,W) and truexyzobj_all (N,6), i.e.,(x1,y1,z1,x2,y2,z2)
    """
    def __init__(self,folderpath_img,dx=3,dy=2,dz=4,nx=11,ny=11,nz=11,scale_z=False):
        #dx,dy,dz (all integer,>=0 by construction) are shift of second object relative to first object in unit of grid points
        #nx,ny,nz are full dataset's num of grid points along x,y,z
        super(FSdataset_twopoints,self).__init__()
        self.FS_all=loadmat(folderpath_img)["FS"]#nx*ny*nz,2,4,4 #Full single object dataset
        self.truexyzobj_all=loadmat(folderpath_img)["xyz"]#Full single object dataset,nx*ny*nz,3
        self.nx,self.ny,self.nz=nx,ny,nz
        self.dx,self.dy,self.dz=dx,dy,dz
        self.scale_z=scale_z

    def __len__(self):
        return (self.nx-self.dx)*(self.ny-self.dy)*(self.nz-self.dz)
    def __getitem__(self,index):
        #input index is the index of data point among this dataset (not covering entire xyz sweep volume, but a smaller rectangular shaped 3D space)
        #index_1/2 are the index of the data point of fulldataset (covering entire xyz sweep volume in experiment)
        #index out single object FS and synthesis two object FS according to dx,dy,dz
        coord_1=np.unravel_index(index, (self.nx-self.dx,self.ny-self.dy,self.nz-self.dz),order='F')#coord (x,y,z) of tensor elements
        coord_2=tuple(map(add, coord_1, (self.dx,self.dy,self.dz)))
        index_1=np.ravel_multi_index(coord_1, (self.nx,self.ny,self.nz),order='F')
        index_2=np.ravel_multi_index(coord_2, (self.nx,self.ny,self.nz),order='F')
        xyz_1=self.truexyzobj_all[index_1]
        xyz_2=self.truexyzobj_all[index_2]
        FS=self.FS_all[index_1]+self.FS_all[index_2] #synthesize FS of two object by addding
        truexyzobj_all=np.concatenate((xyz_1,xyz_2)) #()combine two tuples into one, with first 3 elements corresponds to coordinate of the object with smaller x/y/z, later 3 elements corresponds to coordinates of the object with larger x/y/z
        if self.scale_z==True:
            truexyzobj_all[[2,5]]/=33.3

        return {'FS':FS,'xyz':truexyzobj_all} #FS has dimension nF,H,W, xyz has shape (6,),i.e., (x1,y1,z1,x2,y2,z2)
    
class FSdataset_twopoints_multipleshifts(dset):
    def __init__(self,folderpath_img,dx=[1,3],dy=[2,-2],dz=[2,3],nx=11,ny=11,nz=11,scale_z=False):
        #dx,dx,dz are list of length Nshift, meaning Nshift different shifts possibilities
        super(FSdataset_twopoints_multipleshifts,self).__init__()
        self.Nshift=len(dx)
        self.Datasets_singleshifts=[]
        for i in range(self.Nshift):
        #construct a list with containing all single shift datasets
            self.Datasets_singleshifts.append(FSdataset_twopoints(folderpath_img,dx[i],dy[i],dz[i],nx,ny,nz,scale_z=scale_z))
        self.Dataset_combined=ConcatDataset(self.Datasets_singleshifts)
        
    def __len__(self):
        return len(self.Dataset_combined)
                   
    def __getitem__(self,index):
        return self.Dataset_combined[index]



class FSdataset_twopoints_v2(dset):
    """
    New implementation for general shift possibility (single possibility), the old implementation's shift possibility is not general (second object must has positive shifts along x,y,z)
    
    The data should has two matrix, FS and xyz. FS has dimension N,nF,H(y),W(x), xyz has dimension N,3. Where 3 columns are coordinates of object x,y,z respectively. The N should increase along x first, then y, then z.  

    return a dictionary containing FS (N,nF,H,W) and truexyzobj_all (N,6), i.e.,(x1,y1,z1,x2,y2,z2)
    """
    def __init__(self,folderpath_img,relative_pos,nx=11,ny=11,nz=11,scale_z=False):
        """
        relative_pos: a list with 3D tuple (dx,dy,dz) as list element, where dx,dy,dz specify the relative shifts (all >=0) w.r.t to the bounding box's top left corner ('C') (smallest x,y,z points of the bounding box) . The number of 3D tuple should be same as the number of obj points. (2 for this two points dataset)
        """
        super(FSdataset_twopoints_v2,self).__init__()
        self.FS_all=loadmat(folderpath_img)["FS"]#nx*ny*nz,2,4,4 #Full single object dataset
        self.truexyzobj_all=loadmat(folderpath_img)["xyz"]#Full single object dataset,nx*ny*nz,3
        self.scale_z=scale_z
        self.nx,self.ny,self.nz=nx,ny,nz
        self.box_wx,self.box_wy,self.box_wz=max(relative_pos[0][0],relative_pos[1][0]),max(relative_pos[0][1],relative_pos[1][1]),max(relative_pos[0][2],relative_pos[1][2]) #calculate the widths of box along x,y,z
        self.relative_pos=relative_pos
        self.dx1,self.dy1,self.dz1=relative_pos[0] #relative shifts of obj 1 w.r.t 'C'
        self.dx2,self.dy2,self.dz2=relative_pos[1] #relative shifts of obj 2 w.r.t 'C'


    def __len__(self):
        return (self.nx-self.box_wx)*(self.ny-self.box_wy)*(self.nz-self.box_wz)

    def __getitem__(self, index):
        """
        index: the index of data point in [0,len(dataset)-1] among this dataset. Also index of the box, where the box index to defined arbitarily to be increasing along x first.
        """
        coord_C=np.unravel_index(index, (self.nx-self.box_wx,self.ny-self.box_wy,self.nz-self.box_wz),order='F') #calculate the coordinate of the top left corner ('C') of the bounding box.
        coord_1=tuple(map(add, coord_C, (self.dx1,self.dy1,self.dz1))) # coordinate of 1st obj
        coord_2=tuple(map(add, coord_C, (self.dx2,self.dy2,self.dz2))) # coordinate of 2nd obj
        index_1=np.ravel_multi_index(coord_1, (self.nx,self.ny,self.nz),order='F')
        index_2=np.ravel_multi_index(coord_2, (self.nx,self.ny,self.nz),order='F')
        xyz_1=self.truexyzobj_all[index_1]
        xyz_2=self.truexyzobj_all[index_2]
        FS=self.FS_all[index_1]+self.FS_all[index_2] #synthesize FS of two object by addding
        truexyzobj_all=np.concatenate((xyz_1,xyz_2)) #combine two tuples into one
        if self.scale_z==True:
            truexyzobj_all[[2,5]]/=33.3
        return {'FS':FS,'xyz':truexyzobj_all} #FS has dimension nF,H,W, xyz has shape (6,),i.e., (x1,y1,z1,x2,y2,z2)

    @classmethod
    def test(cls,show=False):
        data_path='E:/Academic/MATLAB/Dehui_FS/2018_11_03 data/18_11_03sweepxyz'
        relative_pos=[(1,0,2),(2,1,0)]
        dataset=cls(data_path,relative_pos,nx=11,ny=11,nz=11,scale_z=False)
        if show == True:
            # display dynamically the datapoints, not runnable in jupyter notebook.
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(len(dataset)):
                ax.scatter(*dataset[i]['xyz'][0:3])
                ax.scatter(*dataset[i]['xyz'][3:6])
                ax.set_xlim3d(-0.3,0.3)
                ax.set_ylim3d(-0.3,0.3)
                ax.set_zlim3d(-10,10)
                plt.show(block=False)
                plt.pause(0.1)
        return dataset

class FSdataset_threepoints_v2(dset):
    """
    New implementation for general shift possibility (single possibility), the old implementation's shift possibility is not general (second object must has positive shifts along x,y,z)
    
    The data should has two matrix, FS and xyz. FS has dimension N,nF,H(y),W(x), xyz has dimension N,3. Where 3 columns are coordinates of object x,y,z respectively. The N should increase along x first, then y, then z.  

    return a dictionary containing FS (N,nF,H,W) and truexyzobj_all (N,9), i.e.,(x1,y1,z1,x2,y2,z2,x3,y3,z3)
    """
    def __init__(self,folderpath_img,relative_pos,nx=11,ny=11,nz=11,scale_z=False):
        """
        relative_pos: a list with 3D tuple (dx,dy,dz) as list element, where dx,dy,dz specify the relative shifts (all >=0) w.r.t to the bounding box's top left corner (smallest x,y,z points of the bounding box) . The number of 3D tuple should be same as the number of obj points. (3 for this two points dataset)
        """
        super(FSdataset_threepoints_v2,self).__init__()
        self.FS_all=loadmat(folderpath_img)["FS"]#nx*ny*nz,2,4,4 #Full single object dataset
        self.truexyzobj_all=loadmat(folderpath_img)["xyz"]#Full single object dataset,nx*ny*nz,3
        self.scale_z=scale_z
        self.nx,self.ny,self.nz=nx,ny,nz
        self.box_wx,self.box_wy,self.box_wz=max(relative_pos[0][0],relative_pos[1][0],relative_pos[2][0]),max(relative_pos[0][1],relative_pos[1][1],relative_pos[2][1]),max(relative_pos[0][2],relative_pos[1][2],relative_pos[2][2]) #calculate the widths of box along x,y,z
        self.relative_pos=relative_pos
        self.dx1,self.dy1,self.dz1=relative_pos[0] #relative shifts of obj 1 w.r.t 'C'
        self.dx2,self.dy2,self.dz2=relative_pos[1] #relative shifts of obj 2 w.r.t 'C'
        self.dx3,self.dy3,self.dz3=relative_pos[2] #relative shifts of obj 3 w.r.t 'C'

    def __len__(self):
        return (self.nx-self.box_wx)*(self.ny-self.box_wy)*(self.nz-self.box_wz)

    def __getitem__(self, index):
        """
        index: the index of data point in [0,len(dataset)-1] among this dataset. Also index of the box, where the box index to defined arbitarily to be increasing along x first.
        """
        coord_C=np.unravel_index(index, (self.nx-self.box_wx,self.ny-self.box_wy,self.nz-self.box_wz),order='F') #calculate the coordinate of the top left corner ('C') of the bounding box.
        coord_1=tuple(map(add, coord_C, (self.dx1,self.dy1,self.dz1))) # coordinate of 1st obj
        coord_2=tuple(map(add, coord_C, (self.dx2,self.dy2,self.dz2))) # coordinate of 2nd obj
        coord_3=tuple(map(add, coord_C, (self.dx3,self.dy3,self.dz3))) # coordinate of 3nd obj
        index_1=np.ravel_multi_index(coord_1, (self.nx,self.ny,self.nz),order='F')
        index_2=np.ravel_multi_index(coord_2, (self.nx,self.ny,self.nz),order='F')
        index_3=np.ravel_multi_index(coord_3, (self.nx,self.ny,self.nz),order='F')
        xyz_1=self.truexyzobj_all[index_1]
        xyz_2=self.truexyzobj_all[index_2]
        xyz_3=self.truexyzobj_all[index_3]
        FS=self.FS_all[index_1]+self.FS_all[index_2]+self.FS_all[index_3] #synthesize FS of three objects by addding
        truexyzobj_all=np.concatenate((xyz_1,xyz_2,xyz_3)) #combine two tuples into one
        if self.scale_z==True:
            truexyzobj_all[[2,5,8]]/=33.3
        return {'FS':FS,'xyz':truexyzobj_all} #FS has dimension nF,H,W, xyz has shape (9,),i.e., (x1,y1,z1,x2,y2,z2,x3,y3,z3)

    @classmethod
    def test(cls):
        data_path='E:/Academic/MATLAB/Dehui_FS/2018_11_03 data/18_11_03sweepxyz'
        relative_pos=[(1,0,2),(2,1,0),[3,2,1]]
        dataset=cls(data_path,relative_pos,nx=11,ny=11,nz=11,scale_z=False)
        return dataset
class FSdataset_twopoints_multipleshifts_v2(dset):
    """Generate two points dataset with multiple shifts"""
    def __init__(self,folderpath_img,relative_pos_list,nx=11,ny=11,nz=11,scale_z=False):
        """
        relative_pos_list is a list of length Nshift, meaning Nshift different shifts possibilities,
        where each list element a also a list: relative_pos
        """
        super(FSdataset_twopoints_multipleshifts_v2,self).__init__()
        self.Nshift=len(relative_pos_list)
        self.Datasets_singleshifts=[]
        for i in range(self.Nshift):
        #construct a list with containing all single shift datasets
            self.Datasets_singleshifts.append(FSdataset_twopoints_v2(folderpath_img,relative_pos_list[i],nx,ny,nz,scale_z=scale_z))
        self.Dataset_combined=ConcatDataset(self.Datasets_singleshifts)

    def __len__(self):
        return len(self.Dataset_combined)
                   
    def __getitem__(self,index):
        return self.Dataset_combined[index]

    @classmethod
    def test(cls):
        data_path='E:/Academic/MATLAB/Dehui_FS/2018_11_03 data/18_11_03sweepxyz'
        relative_pos_list=[[(1,0,2),(2,1,0)],[(3,0,1),(2,0,0)]] #for case of 2 shifts
        dataset=cls(data_path,relative_pos_list,nx=11,ny=11,nz=11,scale_z=False)
        return dataset
class FSdataset_threepoints_multipleshifts_v2(dset):
    """Generate three points dataset with multiple shifts"""
    def __init__(self,folderpath_img,relative_pos_list,nx=11,ny=11,nz=11,scale_z=False):
        """
        relative_pos_list is a list of length Nshift, meaning Nshift different shifts possibilities,
        where each list element a also a list: relative_pos
        """
        super(FSdataset_threepoints_multipleshifts_v2,self).__init__()
        self.Nshift=len(relative_pos_list)
        self.Datasets_singleshifts=[]
        for i in range(self.Nshift):
        #construct a list with containing all single shift datasets
            self.Datasets_singleshifts.append(FSdataset_threepoints_v2(folderpath_img,relative_pos_list[i],nx,ny,nz,scale_z=scale_z))
        self.Dataset_combined=ConcatDataset(self.Datasets_singleshifts)
        
    def __len__(self):
        return len(self.Dataset_combined)
                   
    def __getitem__(self,index):
        return self.Dataset_combined[index]

    @classmethod
    def test(cls):
        data_path='E:/Academic/MATLAB/Dehui_FS/2018_11_03 data/18_11_03sweepxyz'
        relative_pos_list=[[(1,0,2),(2,1,0),(1,1,0)],[(3,0,1),(2,0,0),(1,0,1)]] #for case of 2 shifts
        dataset=cls(data_path,relative_pos_list,nx=11,ny=11,nz=11,scale_z=False)
        return dataset
class depthnet3_Nobj(nn.Module):
    ##return coordinates values along one dimension for all objs
    def __init__(self,input_dim=32,num_features=16,Nobj=2):
        super(depthnet3_Nobj, self).__init__()
        self.Linear1=nn.Linear(input_dim, num_features,bias=True)
        self.Linear2=nn.Linear(num_features,Nobj,bias=True)
        
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=self.Linear2(x)

        return x

class depthnet4_Nobj(nn.Module):
    ##return coordinates values along one dimension for all objs
    def __init__(self,input_dim=32,num_features=16,Nobj=2):
        super(depthnet4_Nobj, self).__init__()
        self.Linear1=nn.Linear(input_dim, num_features,bias=True)
        self.Linear2=nn.Linear(num_features,num_features,bias=True)
        self.Linear3=nn.Linear(num_features,Nobj,bias=True)
      
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=F.relu(self.Linear2(x))
        x=self.Linear3(x)

        return x 
class depthnet3_Nobj_xyz(nn.Module):
    #predicting the xyz of two objects at once
    #output is assumed to have form N,6. The 6 columns are (x1,y1,z1,x2,y2,z2...xN,yN,zN)
    def __init__(self,input_dim=32,num_features=16,Nobj=2):
        super(depthnet3_Nobj_xyz, self).__init__()
        self.Linear1=nn.Linear(input_dim, num_features,bias=True)
        self.Linear2=nn.Linear(num_features,Nobj*3,bias=True)
        
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=self.Linear2(x)

        return x    
    
class depthnet4_Nobj_xyz(nn.Module):
    #predicting the xyz of two objects at once
    #output is assumed to have form N,6. The 6 columns are (x1,y1,z1,x2,y2,z2...xN,yN,zN)
    def __init__(self,input_dim=32,num_features=16,Nobj=2):
        super(depthnet4_Nobj_xyz, self).__init__()
        self.Linear1=nn.Linear(input_dim, num_features,bias=True)
        self.Linear2=nn.Linear(num_features,num_features,bias=True)
        self.Linear3=nn.Linear(num_features,Nobj*3,bias=True)
      
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=F.relu(self.Linear2(x))
        x=self.Linear3(x)

        return x    
    
class depthnet5_Nobj_xyz(nn.Module):
    #predicting the xyz of two objects at once
    #output is assumed to have form N,6. The 6 columns are (x1,y1,z1,x2,y2,z2...xN,yN,zN)
    def __init__(self,input_dim=32,num_features=16,Nobj=2):
        super(depthnet5_Nobj_xyz, self).__init__()
        self.Linear1=nn.Linear(input_dim, num_features,bias=True)
        self.Linear2=nn.Linear(num_features,num_features,bias=True)
        self.Linear3=nn.Linear(num_features,num_features,bias=True)
        self.Linear4=nn.Linear(num_features,Nobj*3,bias=True)
      
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=F.relu(self.Linear2(x))
        x=F.relu(self.Linear3(x))
        x=self.Linear4(x)

        return x    
    
class depthnet6_Nobj_xyz(nn.Module):
    #predicting the xyz of two objects at once
    #output is assumed to have form N,6. The 6 columns are (x1,y1,z1,x2,y2,z2...xN,yN,zN)
    def __init__(self,input_dim=32,num_features=16):
        super(depthnet6_Nobj_xyz, self).__init__()
        self.Linear1=nn.Linear(input_dim, num_features,bias=True)
        self.Linear2=nn.Linear(num_features,num_features,bias=True)
        self.Linear3=nn.Linear(num_features,num_features,bias=True)
        self.Linear4=nn.Linear(num_features,num_features,bias=True)
        self.Linear5=nn.Linear(num_features,Nobj*3,bias=True)
      
        
    def forward(self,x):
        N,nF,H,W=x.shape
        x=F.relu(self.Linear1(x.view(N,-1)))
        x=F.relu(self.Linear2(x))
        x=F.relu(self.Linear3(x))
        x=F.relu(self.Linear4(x))
        x=self.Linear5(x)

        return x    
    


class Set_loss_2obj(nn.Module):
    def __init__(self):
        super(Set_loss_2obj, self).__init__()
        
    def forward(self,coord_1_pred,coord_2_pred,coord_1_true,coord_2_true):
        """
        coord_1_pred,coord_2_pred,coord_1_true,coord_2_true: NX3 tensors, each row is (x,y,z)
        The function calculate the order indenpendent loss by choosing the order with smaller difference between the true and pred.
        (can be extended to more than two objs, though computational expensive)
        Return loss of shape (1,)
        """
        loss1=torch.sum((coord_1_pred-coord_1_true)**2+(coord_2_pred-coord_2_true)**2,dim=1) #size(N,1)
        loss2=torch.sum((coord_1_pred-coord_2_true)**2+(coord_2_pred-coord_1_true)**2,dim=1) #size(N,1)
        loss=torch.cat((torch.unsqueeze(loss1,dim=1),torch.unsqueeze(loss2,dim=1)),dim=1) #size(N,2)
        loss,_=torch.min(loss,dim=1) #size(N,1)
        return torch.sum(loss)/loss.shape[0]

class Set_loss_3obj(nn.Module):
    def __init__(self):
        super(Set_loss_3obj, self).__init__()
        
    def forward(self,coord_1_pred,coord_2_pred,coord_3_pred,coord_1_true,coord_2_true,coord_3_true):
        """
        coord_*_pred,coord_*_true,: NX3 tensors, each row is (x,y,z)
        The function calculate the order indenpendent loss by choosing the order with smaller difference between the true and pred.
        (can be extended to more than two objs, though computational expensive)
        Return loss of shape (1,)
        """
        loss1=torch.sum((coord_1_pred-coord_1_true)**2+(coord_2_pred-coord_2_true)**2+(coord_3_pred-coord_3_true)**2,dim=1) #size(N,)
        loss2=torch.sum((coord_1_pred-coord_1_true)**2+(coord_2_pred-coord_3_true)**2+(coord_3_pred-coord_2_true)**2,dim=1) #size(N,)
        loss3=torch.sum((coord_1_pred-coord_2_true)**2+(coord_2_pred-coord_1_true)**2+(coord_3_pred-coord_3_true)**2,dim=1) #size(N,)
        loss4=torch.sum((coord_1_pred-coord_2_true)**2+(coord_2_pred-coord_3_true)**2+(coord_3_pred-coord_1_true)**2,dim=1) #size(N,)
        loss5=torch.sum((coord_1_pred-coord_3_true)**2+(coord_2_pred-coord_1_true)**2+(coord_3_pred-coord_2_true)**2,dim=1) #size(N,)
        loss6=torch.sum((coord_1_pred-coord_3_true)**2+(coord_2_pred-coord_2_true)**2+(coord_3_pred-coord_1_true)**2,dim=1) #size(N,)
        loss=torch.cat((torch.unsqueeze(loss1,dim=1),torch.unsqueeze(loss2,dim=1),torch.unsqueeze(loss3,dim=1),torch.unsqueeze(loss4,dim=1),torch.unsqueeze(loss5,dim=1),torch.unsqueeze(loss6,dim=1)),dim=1) #size(N,6)
        loss,_=torch.min(loss,dim=1) #size(N,1)
        return torch.sum(loss)/loss.shape[0]


    
def genRotation_data_idx_single_orientation(x,y,box_wx,box_wy,nx=11,ny=11,nz=11,offset = 0):
    """
    For generating the flattened data idx for the single orientation dataset FSdataset_twopoints_v2,where each sample has to be a point pair in xy plane, with no z component. 
    Input:
    x,y index of the trajectory [0, max]
    box_wx,box_wy：width of the two point bounding box, note these should be divisible by 2
    nx,ny,nz: spatial grid resolution
    offset: 0 by default, only non-zero when used in genRotation_data_idx_multi_orientation
    Output: 
    flattened index into the dataset
    """
    number_of_data = (nx-box_wx)*(ny-box_wy)*nz
    idx = np.ravel_multi_index([x-int(box_wx/2),y-int(box_wy/2)],[nx-box_wx,ny-box_wy], order = 'F')
    return np.arange(idx,number_of_data,(nx-box_wx)*(ny-box_wy)) + offset

def genRotation_data_idx_multi_orientation(x,y,box_wx,box_wy,nx=11,ny=11,nz=11,single_orientation_ds_list=None,rotating_idx=None):
    """
    For generating the flattened data idx for the concatenated orientation dataset (by concatenating FSdataset_twopoints_v2 having different orientations),where each sample has to be a point pair in xy plane, with no z component. 
    Input:
    x,y index of the trajectory [0, max]
    box_wx,box_wy：width of the two point bounding box, note these should be divisible by 2
    nx,ny,nz: spatial grid resolution
    single_orientation_ds_list: [ds1,ds2,....], where each ds is a single orientation two point dataset created from FSdataset_twopoints_v2
    rotating_idx: list having length of the trajectory depth, where each element of the list specify which orientation dataset to use for each depth z. 
                    The element specify orientation in the order of increaseing z.
    Output: 
    flattened index into the dataset
    """
    numOrientation = len(single_orientation_ds_list)
    index_list = []
    offset = 0
    idx = np.zeros(len(rotating_idx))
    #generating index of at all orientation at all depths z
    for i in range(numOrientation):
        index_list.append(genRotation_data_idx_single_orientation(x,y,box_wx[i],box_wy[i],nx=11,ny=11,nz=11,offset = offset))
        offset += len(single_orientation_ds_list[i]) 

    for counter, value in enumerate(rotating_idx):
        idx[counter] = index_list[value][counter]
    return idx.astype(int).tolist()
