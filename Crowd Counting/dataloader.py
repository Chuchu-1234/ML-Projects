from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


class CrowdDataset(Dataset):
    
    def __init__(self,img_root,groundtruth_dmap_root,gt_downsample=1):
        
        self.img_root=img_root
        self.groundtruth_dmap_root=groundtruth_dmap_root
        self.gt_downsample=gt_downsample

        self.img_names=[filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root,filename))]
        self.n_samples=len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        img=plt.imread(os.path.join(self.img_root,img_name))
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        groundtruth_dmap=np.load(os.path.join(self.groundtruth_dmap_root,img_name.replace('.jpg','.npy')))
        print(gt_dmap.sum())
        if self.gt_downsample>1: 
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            img=img.transpose((2,0,1)) 
            groundtruth_dmap=cv2.resize(groundtruth_dmap,(ds_cols,ds_rows))
            groundtruth_dmap=groundtruth_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample

        img_tensor=torch.tensor(img,dtype=torch.float)
        groundtruth_tensor=torch.tensor(groundtruth_dmap,dtype=torch.float)
        #print(groundtruth_tensor.sum())

        return img_tensor,groundtruth_tensor