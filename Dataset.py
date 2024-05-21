import os
import sys
import numpy as np
from PIL import Image
import cv2
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import GaussianBlur

from utils.LFProcess import *
from utils.ImgProcess import *


#Data for training VAE
class VAETrainData(Dataset):
    def __init__(self,data_root):
        super(VAETrainData,self).__init__()
        self.data_root=data_root
        self.path=[]
        for item in os.listdir(data_root):
            if not item.startswith('.'):
                for i in range(49):
                    self.path.append([os.path.join(data_root,item),i])
    
    def __getitem__(self,i):
        path=self.path[i][0]
        idx=self.path[i][1]
        img_mp = np.array(Image.open(path))

        img_mp = 2*(img_mp/255)-1   
        img_sais=MP2SAIs(img_mp,ah=7,aw=7)  #[49,H,W,3]
        img=img_sais[idx] #[H,W,3]

        img = torch.from_numpy(img).float().permute(2,0,1)  #[3,H,W]
        
        return img
    
    def __len__(self):
        return len(self.path)
    
    
class VAETestData(Dataset):
    def __init__(self,data_root):
        super(VAETestData,self).__init__()
        self.data_root=data_root
        self.path=[]
        for item in os.listdir(data_root):
            if not item.startswith('.'):
                self.path.append(os.path.join(data_root,item))
        
    def __getitem__(self,i):
        img_mp = np.array(Image.open(self.path[i]))
        
        img_mp = 2*(img_mp/255)-1   #[H,W,3]
        img_sais=MP2SAIs(img_mp,ah=7,aw=7)  #[49,H,W,3]
        img=img_sais[24] #[H,W,3]
        
        img = torch.from_numpy(img).float().permute(2,0,1)  #[3,H,W]
        
        return img
    
    def __len__(self):
        return len(self.path)


#Data for single image 
class SingleSynData(Dataset):
    def __init__(self,data_root,mode):
        super(SingleSynData,self).__init__()
        self.gt_list=[]
        self.mode=mode
        for item in os.listdir(data_root):
            if not item.startswith('.'):
                gt_mp=np.array(Image.open(os.path.join(data_root,item)))
                self.gt_list.append(gt_mp)
                        
    def __getitem__(self,i):
        gt_mp = self.gt_list[i]

        #Select parameters randomly
        v_scale=np.random.uniform(0.05,0.2)
        k=np.random.uniform(1,4)
        inp_mp=LowLF_syn(gt_mp,v_scale,k)
        
        gt_sais=MP2SAIs(gt_mp,ah=7,aw=7)  #[49,H,W,3]
        inp_sais=MP2SAIs(inp_mp,ah=7,aw=7)  #[49,H,W,3]
        if self.mode=='train':
            idx=np.random.randint(low=0,high=gt_sais.shape[0])
        else:
            idx=gt_sais.shape[0]//2
        gt=gt_sais[idx]  #[H,W,3]
        inp=inp_sais[idx]  #[H,W,3]

        inp=inp/255
        rgb_max=np.max(inp,axis=(0,1))
        inp_c=inp/rgb_max  #[H,W,3]
            
        inp = 2*inp-1   
        inp_c = 2*inp_c-1  
        inp = np.concatenate((inp,inp_c),axis=-1)  #[H,W,6]
        gt = 2*(gt/255)-1   
        
        inp = torch.from_numpy(inp).float().permute(2,0,1)  #[6,H,W]
        gt = torch.from_numpy(gt).float().permute(2,0,1)  #[3,H,W]
        
        return inp, gt
    
    def __len__(self):
        return len(self.gt_list)
        
    
class SingleL3FData(Dataset):
    def __init__(self,data_root):
        super(SingleL3FData,self).__init__()
        self.data_root=data_root
        self.gt_list=[]
        self.inp_list=[]
        
        for item in os.listdir(os.path.join(data_root,'GT')):
            if not item.startswith('.'):
                gt_mp=np.array(Image.open(os.path.join(data_root,'GT',item)))
                inp_mp=np.array(Image.open(os.path.join(data_root,'Inp',item)))
                self.gt_list.append(gt_mp)
                self.inp_list.append(inp_mp)
        
    def __getitem__(self,i):
        gt_mp = self.gt_list[i]
        inp_mp = self.inp_list[i]

        gt_sais=MP2SAIs(gt_mp,ah=7,aw=7)  #[49,H,W,3]
        inp_sais=MP2SAIs(inp_mp,ah=7,aw=7)  #[49,H,W,3]
        idx=np.random.randint(low=0,high=gt_sais.shape[0])
        gt=gt_sais[idx]  #[H,W,3]
        inp=inp_sais[idx]  #[H,W,3]

        inp=inp/255
        rgb_max=np.max(inp,axis=(0,1))
        inp_c=inp/rgb_max  #[H,W,3]
            
        inp = 2*inp-1   
        inp_c = 2*inp_c-1  
        inp = np.concatenate((inp,inp_c),axis=-1)  #[H,W,6]
        gt = 2*(gt/255)-1   
        
        inp = torch.from_numpy(inp).float().permute(2,0,1)  #[6,H,W]
        gt = torch.from_numpy(gt).float().permute(2,0,1)  #[3,H,W]
        
        return inp, gt
    
    def __len__(self):
        return len(self.gt_list)


#Data for LF image 
class LF3x3SynData(Dataset):
    def __init__(self,data_root,mode):
        super(LF3x3SynData,self).__init__()
        self.gt_list=[]
        self.mode=mode
        for item in os.listdir(data_root):
            if not item.startswith('.'):
                gt_mp=np.array(Image.open(os.path.join(data_root,item)))
                self.gt_list.append(gt_mp)
    
    def __getitem__(self,i):
        gt_mp = self.gt_list[i]
        
        #Select parameters randomly
        v_scale=np.random.uniform(0.05,0.2)
        k=np.random.uniform(1,4)
        inp_mp=LowLF_syn(gt_mp,v_scale,k)

        gt_sais=MP2SAIs(gt_mp,ah=7,aw=7)  #[49,H,W,3]
        inp_sais=MP2SAIs(inp_mp,ah=7,aw=7)  #[49,H,W,3]
        
        if self.mode=='train':
            ind_c=random.choice([8,9,10,11,12,15,16,17,18,19,22,23,24,25,26,29,30,31,32,33,36,37,38,39,40])
        else:
            ind_c=24
        inds=[ind_c-8,ind_c-7,ind_c-6,ind_c-1,ind_c,ind_c+1,ind_c+6,ind_c+7,ind_c+8]
            
        gt_sais=gt_sais[inds]  #[9,H,W,3]
        inp_sais=inp_sais[inds]  #[9,H,W,3]  

        inp_sais=inp_sais/255
        rgb_max=np.max(inp_sais,axis=(1,2),keepdims=True)  #[9,1,1,3]
        inp_c=inp_sais/rgb_max  #[9,H,W,3]

        inp_sais = 2*inp_sais-1   
        inp_c = 2*inp_c-1  
        inp_sais = np.concatenate((inp_sais,inp_c),axis=-1)  #[9,H,W,6]
        gt_sais = 2*(gt_sais/255)-1

        inp_sais=torch.from_numpy(inp_sais).float().permute(0,3,1,2)  #[9,6,H,W]
        gt_sais=torch.from_numpy(gt_sais).float().permute(0,3,1,2)  #[9,3,H,W]
        
        return inp_sais,gt_sais
    
    def __len__(self):
        return len(self.gt_list)


class LF3x3L3FData(Dataset):
    def __init__(self,data_root):
        super(LF3x3L3FData,self).__init__()
        self.data_root=data_root
        self.gt_list=[]
        self.inp_list=[]
        
        for item in os.listdir(os.path.join(data_root,'GT')):
            if not item.startswith('.'):
                gt_mp=np.array(Image.open(os.path.join(data_root,'GT',item)))
                inp_mp=np.array(Image.open(os.path.join(data_root,'Inp',item)))
                self.gt_list.append(gt_mp)
                self.inp_list.append(inp_mp)

    def __getitem__(self,i):
        gt_mp = self.gt_list[i]
        inp_mp = self.inp_list[i]

        gt_sais=MP2SAIs(gt_mp,ah=7,aw=7)  #[49,H,W,3]
        inp_sais=MP2SAIs(inp_mp,ah=7,aw=7)  #[49,H,W,3] 

        ind_c=random.choice([8,9,10,11,12,15,16,17,18,19,22,23,24,25,26,29,30,31,32,33,36,37,38,39,40])
        inds=[ind_c-8,ind_c-7,ind_c-6,ind_c-1,ind_c,ind_c+1,ind_c+6,ind_c+7,ind_c+8]

        gt_sais=gt_sais[inds]  #[9,H,W,3]
        inp_sais=inp_sais[inds]  #[9,H,W,3]  

        inp_sais=inp_sais/255
        rgb_max=np.max(inp_sais,axis=(1,2),keepdims=True)  #[9,1,1,3]
        inp_c=inp_sais/rgb_max  #[9,H,W,3]

        inp_sais = 2*inp_sais-1   
        inp_c = 2*inp_c-1  
        inp_sais = np.concatenate((inp_sais,inp_c),axis=-1)  #[9,H,W,6]
        gt_sais = 2*(gt_sais/255)-1

        inp_sais=torch.from_numpy(inp_sais).float().permute(0,3,1,2)  #[9,6,H,W]
        gt_sais=torch.from_numpy(gt_sais).float().permute(0,3,1,2)  #[9,3,H,W]

        return inp_sais,gt_sais
    
    def __len__(self):
        return len(self.gt_list)