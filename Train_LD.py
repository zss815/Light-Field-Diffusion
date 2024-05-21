import os
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset

from Dataset import SingleSynData, SingleL3FData
from tools import *
from Loss import RestoreLoss

from utils.metrics import PSNR, SSIM
from networks.LDUNet import LDUNet
from networks.VAE import VAE


def train(args):
    epoch_dict,psnr_dict,ssim_dict={},{},{}
    for i in range(1,args.save_num+1):
        epoch_dict[i]=0
        psnr_dict[i]=0
        ssim_dict[i]=0
    best_psnr=0
    best_ssim=0
    best_epoch=0
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root,exist_ok=True)
            
    train_set1=SingleSynData(data_root='/mnt/disk/LFLL/LFReal/Train', mode='train')
    train_set2=SingleL3FData(data_root='/mnt/disk/LFLL/L3F/Train')
    train_set=ConcatDataset([train_set1,train_set2])
    
    test_set=SingleSynData(data_root='/mnt/disk/LFLL/LFReal/Test', mode='test')
    test_num=len(test_set)

    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_set,batch_size=args.batch_size,shuffle=False)
    
    model = LDUNet(in_channels=4,con_channels=6,base_channels=32,t1_channels=32,t2_channels=64,pool_sizes=[16,8,4,2])
    print('model parameters: {}'.format(sum(param.numel() for param in model.parameters())))
    model.cuda() 
    #model=torch.nn.DataParallel(model)

    #Load VAE
    with torch.no_grad():
        VAE_model = VAE(in_channels=3,z_channels=4,hid_channels=[64,128,256,512],strides=[1,2,2,2])
        VAE_model.cuda()
        VAE_model.load_state_dict(torch.load(args.VAE_path))
        VAE_model.eval()
        for param in VAE_model.parameters():
            param.requires_grad=False

    Loss_img=RestoreLoss()
    Loss_l1=torch.nn.L1Loss()

    if args.pre_train:
        model.load_state_dict(torch.load(args.model_path))

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr_init)
    
    num_diffusion_steps=1000
    num_sampling_steps=25
    betas = beta_schedule(schedule='linear', beta_start=0.0001, beta_end=0.02, num_diffusion_steps=num_diffusion_steps)
    betas = torch.from_numpy(betas).float().cuda()
    alphas = (1-betas).cumprod(dim=0)
    skip = num_diffusion_steps // num_sampling_steps
    seq = torch.arange(0, num_diffusion_steps, skip)
    
    n=0
    for epoch in range(args.max_epoch):
        model.train()
        for idx,(inp,gt) in enumerate(train_loader):
            inp,gt=inp.cuda(),gt.cuda() #[B,3,H,W]
            B=inp.shape[0]

            r=0.8
            t1 = torch.randint(low=0, high=int(num_diffusion_steps*r), size=(B//2,)).cuda()
            t2 = torch.randint(low=int(num_diffusion_steps*r), high=num_diffusion_steps, size=(B-B//2,)).cuda()
            t = torch.cat([t1, t2], dim=0)   #[B]
            a = alphas.index_select(0, t).reshape(-1,1,1,1) #[B,1,1,1]

            z_gt=VAE_model.encoder(gt)  #[B,4,h,w]
            e = torch.randn_like(z_gt).cuda()   #[B,4,h,w]
            zt=z_gt*a.sqrt()+e*(1-a).sqrt()  #[B,3,h,w]

            e_pred = model(zt, inp, t) #[B,3,h,w]
            z0 = (zt - e_pred * (1 - a).sqrt()) / a.sqrt()  #[B,3,h,w]
            x0 = VAE_model.decoder(z0)  #[B,3,H,W]
            
            e_loss=Loss_l1(e_pred,e)
            x0=inverse_data_transform(x0)
            gt=inverse_data_transform(gt)
            x_loss=Loss_img(x0,gt)
            loss=e_loss+x_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: %i, step: %i, e_loss: %f' %(epoch, idx, e_loss.item()))
            print('')
  
        model.eval()
        with torch.no_grad():
            total_psnr=0
            total_ssim=0
            
            for inp,gt in test_loader:
                inp=inp.cuda()
                z_pred = Sample_LD(inp, seq, model, betas)  #[B,4,h,w]
                pred = VAE_model.decoder(z_pred)  #[B,3,H,W]

                pred=inverse_data_transform(pred)  #[B,3,H,W]
                gt=inverse_data_transform(gt)      #[B,3,H,W]
                pred=pred.cpu().numpy()  
                gt=gt.numpy()
                B=pred.shape[0]
                
                for i in range(B):
                    p=pred[i]
                    g=gt[i]
                    psnr=PSNR(g,p,data_range=1)
                    ssim=SSIM(g,p,data_range=1)
                    total_psnr+=psnr
                    total_ssim+=ssim
                        
            ave_psnr=total_psnr/test_num
            ave_ssim=total_ssim/test_num
            print('Epoch {}, average PSNR {}, SSIM {}'.format(epoch,ave_psnr,ave_ssim))
            print('')
    
        #save models            
        if n<args.save_num:
            n+=1
            torch.save(model.state_dict(),os.path.join(args.save_root,'model%s.pth'%str(n)))
            psnr_dict[n]=ave_psnr
            ssim_dict[n]=ave_ssim
            epoch_dict[n]=epoch
        else:
            if ave_psnr>min(psnr_dict.values()):
                torch.save(model.state_dict(),os.path.join(args.save_root,'model%s.pth'%str(min(psnr_dict,key=lambda x: psnr_dict[x]))))
                epoch_dict[min(psnr_dict,key=lambda x: psnr_dict[x])]=epoch
                ssim_dict[min(psnr_dict,key=lambda x: psnr_dict[x])]=ave_ssim
                psnr_dict[min(psnr_dict,key=lambda x: psnr_dict[x])]=ave_psnr                
        if ave_psnr>best_psnr:
            best_psnr=ave_psnr
            best_ssim=ave_ssim 
            best_epoch=epoch
        print('Best PSNR {}, SSIM {}, epoch {}'.format(best_psnr,best_ssim,best_epoch))
        print('Epoch {}'.format(epoch_dict))
        print('PSNR {}'.format(psnr_dict))
        print('SSIM {}'.format(ssim_dict))
        print('')
        
        
if __name__=='__main__':  
    parser = argparse.ArgumentParser(description='LDUNet')
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr_init', default=5e-4, type=float)
    parser.add_argument('--max_epoch',default=1000000,type=int)
    parser.add_argument('--save_num', default=10, type=int, help='number of saved models')
    parser.add_argument('--VAE_path',default='/mnt/disk/save_models/Diffusion/Lowlight/VAE/model1.pth',type=str)
    parser.add_argument('--save_root',default='/mnt/disk/save_models/Diffusion/Lowlight/LDUNet',type=str)
    parser.add_argument('--pre_train',default=False,type=bool)
    parser.add_argument('--model_path',default='/mnt/disk/save_models/Diffusion/Lowlight/LDUNet/model2.pth',type=str)
    args = parser.parse_known_args()[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    train(args)