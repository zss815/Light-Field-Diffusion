import torch
import numpy as np
import cv2
from PIL import Image
import time
import os
import sys

from tools import *
from networks.LDUNet import LDUNet
from networks.VAE import VAE
from networks.LFRefineNet import LFRefineNet
from utils.LFProcess import *
from utils.ImgProcess import *
from utils.metrics import PSNR, SSIM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.lpips import LPIPS
lpips = LPIPS('alex', '0.1')
lpips.cuda()


#Load model
model = LDUNet(in_channels=4,con_channels=6,base_channels=32,t1_channels=32,t2_channels=64,pool_sizes=[16,8,4,2]) 
print('Diffusion model parameters: {}'.format(sum(param.numel() for param in model.parameters())))
model.cuda()
model_path='/mnt/disk/save_models/Diffusion/Lowlight/LDUNet_MVA_ft/model1.pth'
model.load_state_dict(torch.load(model_path)['LD'])
model.eval()

model_refine = LFRefineNet(in_channels=9,out_channels=3,base_channels=16,an=3,patch_size=16)
print('Refinement model parameters: {}'.format(sum(param.numel() for param in model_refine.parameters())))
model_refine.cuda()
model_refine.load_state_dict(torch.load(model_path)['refine'])
model_refine.eval()

VAE_model = VAE(in_channels=3,z_channels=4,hid_channels=[64,128,256,512],strides=[1,2,2,2])
VAE_model.cuda()
VAE_model.load_state_dict(torch.load('/mnt/disk/save_models/Diffusion/Lowlight/VAE/model1.pth'))
VAE_model.eval()

data_root='/mnt/disk/LFLL/LFReal/Test'
#data_root='/mnt/disk/LFLL/L3F/Test'
save_root='/mnt/disk/Results'

num_diffusion_steps=1000
num_sampling_steps=25
betas = beta_schedule(schedule='linear', beta_start=0.0001, beta_end=0.02, num_diffusion_steps=num_diffusion_steps)
betas = torch.from_numpy(betas).float().cuda()
skip = num_diffusion_steps // num_sampling_steps
seq = range(0, num_diffusion_steps, skip)


#Test one sample 
item='1.jpeg'
with torch.no_grad():
    #Synthetic data
    gt_mp = np.array(Image.open(os.path.join(data_root,item)))
    inp_mp=LowLF_syn(gt_mp,v_scale=0.1,k=2)

    #Real data
    # gt_mp = np.array(Image.open(os.path.join(data_root,'GT',item)))
    # inp_mp = np.array(Image.open(os.path.join(data_root,'Inp',item)))
        
    gt_sais=MP2SAIs(gt_mp,ah=7,aw=7)  #[49,H,W,3]
    inp_sais=MP2SAIs(inp_mp,ah=7,aw=7)  #[49,H,W,3]

    inp_sais=inp_sais/255
    rgb_max=np.max(inp_sais,axis=(1,2),keepdims=True)
    inp_c=inp_sais/rgb_max  #[49,H,W,3]
        
    inp_sais = 2*inp_sais-1   
    inp_c = 2*inp_c-1  
    inp_sais = np.concatenate((inp_sais,inp_c),axis=-1)  #[H,W,6]
    gt_sais = gt_sais/255
        
    inp_sais=torch.from_numpy(inp_sais).float().permute(0,3,1,2)  #[N,3,H,W]
    N,H,W=gt_sais.shape[0:3]
    ah=aw=int(N**0.5)

    ind_list=[]   
    ind_c=[8,10,12,22,24,26,36,38,40]
    for ic in ind_c:
        ind_list.append([ic-8,ic-7,ic-6,ic-1,ic,ic+1,ic+6,ic+7,ic+8])

    pred_list=[]
    for ind in ind_list:
        inp_patch=inp_sais[ind].unsqueeze(dim=0).cuda()  #[1,9,3,H,W]
        z_pred = Sample_LDLF(inp_patch, seq, model, betas)  #[9,4,h,w]
        x0 = VAE_model.decoder(z_pred)  #[9,3,H,W]

        inp_re=torch.cat([x0,inp_patch.squeeze(dim=0)],dim=1)  #[9,6,H,W]
        pred = model_refine(inp_re)   #[9,3,H,W]
        pred = inverse_data_transform(pred)  #[9,3,H,W]
        pred=pred.permute(0,2,3,1).cpu().numpy()  #[9,H,W,3]
        pred_list.append(pred)

    array_list=[]
    for pred in pred_list:
        array=SAIs2Array(pred,ah=3,aw=3)  #[3H,3W,3]
        array_list.append(array)

    array1=np.concatenate([array_list[0][:,:2*W],array_list[1],array_list[2][:,W:]],axis=1)  #[3H,7W,3]
    array2=np.concatenate([array_list[3][:,:2*W],array_list[4],array_list[5][:,W:]],axis=1)  #[3H,7W,3]
    array3=np.concatenate([array_list[6][:,:2*W],array_list[7],array_list[8][:,W:]],axis=1)  #[3H,7W,3]
    array=np.concatenate([array1[:2*H,:],array2,array3[H:,:]],axis=0)  #[7H,7W,3]

    epi=EPI(array,ah,aw,is_row=True,idx=2,position=100)
    pred_sais=Array2SAIs(array,H,W,ah,aw)  #[81,H,W,3]

    psnr,ssim=0,0
    for i in range(N):
        pred=pred_sais[i]
        gt=gt_sais[i]
        p=PSNR(gt,pred,data_range=1)
        s=SSIM(gt,pred,data_range=1,channel_first=False)
        psnr+=p
        ssim+=s
    psnr=psnr/N
    ssim=ssim/N

    pred_view = cv2.cvtColor(np.uint8(pred_sais[12]*255), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_root,item.split('.')[0]+'_view.jpeg'),pred_view)
    pred_mp=SAIs2MP(pred_sais,ah,aw,H,W)
    pred_mp = cv2.cvtColor(np.uint8(pred_mp*255), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_root,item.split('.')[0]+'_mp.jpeg'),pred_mp)
    epi = cv2.cvtColor(np.uint8(epi*255), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_root,item.split('.')[0]+'_epi.jpeg'),epi)

    gt_sais=torch.from_numpy(gt_sais).float().permute(0,3,1,2).cuda()
    pred_sais=torch.from_numpy(pred_sais).float().permute(0,3,1,2).cuda()
    lp=lpips(pred_sais,gt_sais)
    lp=lp.cpu().item()/N

    print('PSNR {}, SSIM {}, LPIPS {}'.format(psnr,ssim,lp))
