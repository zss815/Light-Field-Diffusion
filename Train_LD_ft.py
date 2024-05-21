import os
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset

from Dataset import LF3x3SynData, LF3x3L3FData
from tools import *
from Loss import RestoreLoss

from utils.metrics import PSNR, SSIM
from networks.LDUNet import LDUNet
from networks.LFRefineNet import LFRefineNet
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
            
    train_set1=LF3x3SynData(data_root='/mnt/disk/LFLL/LFReal/Train', mode='train')
    train_set2=LF3x3L3FData(data_root='/mnt/disk/LFLL/L3F/Train')
    train_set=ConcatDataset([train_set1,train_set2])
    
    test_set=LF3x3SynData(data_root='/mnt/disk/LFLL/LFReal/Test', mode='test')
    test_num=len(test_set)

    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_set,batch_size=args.batch_size,shuffle=False)
    
    model_LD = LDUNet(in_channels=4,con_channels=6,base_channels=32,t1_channels=32,t2_channels=64,pool_sizes=[16,8,4,2])
    print('Diffusion model parameters: {}'.format(sum(param.numel() for param in model_LD.parameters())))
    model_LD.cuda() 
    model_LD.load_state_dict(torch.load(args.model_path))

    model_refine = LFRefineNet(in_channels=9,out_channels=3,base_channels=16,an=3,patch_size=16)
    print('Refinement model parameters: {}'.format(sum(param.numel() for param in model_refine.parameters())))
    model_refine.cuda()

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

    optimizer = torch.optim.Adam([{'params':model_LD.parameters(),'lr':1e-4},{'params':model_refine.parameters(),'lr':5e-4}])
    
    num_diffusion_steps=1000
    num_sampling_steps=25
    betas = beta_schedule(schedule='linear', beta_start=0.0001, beta_end=0.02, num_diffusion_steps=num_diffusion_steps)
    betas = torch.from_numpy(betas).float().cuda()
    alphas = (1-betas).cumprod(dim=0)
    skip = num_diffusion_steps // num_sampling_steps
    seq = torch.arange(0, num_diffusion_steps, skip)

    n=0
    for epoch in range(args.max_epoch):
        model_LD.train()
        model_refine.train()
        for idx,(inp,gt) in enumerate(train_loader):
            inp,gt=inp.cuda(),gt.cuda()  #[B,A*A,3,H,W]
            B,N,_,H,W=inp.shape
            inp=inp.reshape(B*N,-1,H,W)  #[B*A*A,6,H,W]
            gt=gt.reshape(B*N,-1,H,W)  #[B*A*A,3,H,W]

            t=torch.randint(low=0, high=num_diffusion_steps, size=(B,)).cuda()  #[B]
            a = alphas.index_select(0, t).reshape(-1,1,1,1,1).repeat(1,N,1,1,1).reshape(B*N,1,1,1) #[B*A*A,1,1,1]

            z_gt=VAE_model.encoder(gt)  #[B*A*A,4,h,w]
            c,h,w=z_gt.shape[1:]
            
            #Noise prior
            p_sq=0.8
            ec = torch.randn(B,c,h,w).cuda()  #[B,4,h,w]
            ei = torch.randn(B,N,c,h,w).cuda()*((1-p_sq)**0.5)    #[B,A*A,4,h,w]
            e = ec[:,None,:,:,:].repeat(1,N,1,1,1)*(p_sq**0.5)+ei   #[B,A*A,4,h,w]
            e[:,N//2,:,:,:]= ec
            e=e.reshape(B*N,c,h,w)   #[B*A*A,4,h,w]

            #Model forward process
            zt=z_gt*a.sqrt()+e*(1-a).sqrt()  #[B*A*A,4,h,w]
            e_pred = model_LD(zt, inp, t)  #[B*A*A,4,h,w]

            z0 = (zt - e_pred * (1 - a).sqrt()) / a.sqrt()  #[B*A*A,4,h,w]
            x0 = VAE_model.decoder(z0)  #[B*A*A,3,H,W]

            inp_re=torch.cat([x0,inp],dim=1)  #[B*A*A,9,H,W]
            x1 = model_refine(inp_re)   #[B*A*A,3,H,W]

            #Calculate losses
            e_loss=Loss_l1(e_pred,e)

            e_pred=e_pred.reshape(B,N,c,h,w)  #[B,A*A,4,h,w]
            ec_pred=e_pred[:,N//2,:,:,:]   #[B,4,h,w]
            ec_loss=Loss_l1(ec_pred,ec)

            ep_pred=torch.cat([e_pred[:,:N//2,:,:,:],e_pred[:,(N//2+1):,:,:,:]],dim=1)  #[B,A*A-1,4,h,w]
            ei=torch.cat([ei[:,:N//2,:,:,:],ei[:,(N//2+1):,:,:,:]],dim=1)  #[B,A*A-1,4,h,w]
            ei_pred=ep_pred-ec_pred[:,None,:,:,:].repeat(1,N-1,1,1,1)*(p_sq**0.5)  #[B,A*A-1,4,h,w]
            ei_loss=Loss_l1(ei_pred,ei)
            
            x0=inverse_data_transform(x0)
            gt=inverse_data_transform(gt)
            x0_loss=Loss_img(x0,gt)

            x1=inverse_data_transform(x1)
            x1_loss=Loss_img(x1,gt)
            
            loss=e_loss+ec_loss+ei_loss+x0_loss+x1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: %i, step: %i, e_loss: %f, ec_loss: %f, ei_loss: %f' %(epoch,idx,e_loss.item(),ec_loss.item(),ei_loss.item()))
            print('')
  
        model_LD.eval()
        model_refine.eval()
        with torch.no_grad():
            total_psnr=0
            total_ssim=0
            
            for inp,gt in test_loader:
                inp=inp.cuda()  #[B,A*A,6,H,W]
                B,N,_,H,W=inp.shape
                z_pred = Sample_LDLF(inp, seq, model_LD, betas)  #[B*A*A,4,h,w]
                x0 = VAE_model.decoder(z_pred)  #[B*A*A,3,H,W]

                inp=inp.reshape(B*N,-1,H,W)
                inp_re=torch.cat([x0,inp[:,:3,:,:]],dim=1)  #[B*A*A,6,H,W]
                pred = model_refine(inp_re)   #[B*A*A,3,H,W]

                pred=inverse_data_transform(pred)  #[B*A*A,3,H,W]
                pred=pred.reshape(B,N,-1,H,W).cpu().numpy()   #[B,A*A,3,H,W]
                gt=inverse_data_transform(gt)      #[B,A*A,3,H,W]
                gt=gt.numpy()
                
                for i in range(B):
                    one_psnr,one_ssim=0,0
                    one_pred=pred[i]
                    one_gt=gt[i]
                    for j in range(N):
                        p=one_pred[j]
                        g=one_gt[j]
                        psnr=PSNR(g,p,data_range=1)
                        ssim=SSIM(g,p,data_range=1)
                        one_psnr+=psnr
                        one_ssim+=ssim
                    one_psnr=one_psnr/N
                    one_ssim=one_ssim/N
                    total_psnr+=one_psnr
                    total_ssim+=one_ssim
                            
            ave_psnr=total_psnr/test_num
            ave_ssim=total_ssim/test_num
                
            print('Epoch {}, average PSNR {}, SSIM {}'.format(epoch,ave_psnr,ave_ssim))
            print('')
    
        #save models            
        if n<args.save_num:
            n+=1
            torch.save({'LD':model_LD.state_dict(),'refine':model_refine.state_dict()},os.path.join(args.save_root,'model%s.pth'%str(n)))
            psnr_dict[n]=ave_psnr
            ssim_dict[n]=ave_ssim
            epoch_dict[n]=epoch
        else:
            if ave_psnr>min(psnr_dict.values()):
                torch.save({'LD':model_LD.state_dict(),'refine':model_refine.state_dict()},os.path.join(args.save_root,'model%s.pth'%str(min(psnr_dict,key=lambda x: psnr_dict[x]))))
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
    parser = argparse.ArgumentParser(description='LFLDUNet')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_epoch',default=1000,type=int)
    parser.add_argument('--save_num', default=10, type=int, help='number of saved models')
    parser.add_argument('--VAE_path',default='/mnt/disk/save_models/Diffusion/Lowlight/VAE/model1.pth',type=str)
    parser.add_argument('--model_path',default='/mnt/disk/save_models/Diffusion/Lowlight/LDUNet/model1.pth',type=str)
    parser.add_argument('--save_root',default='/mnt/disk/save_models/Diffusion/Lowlight/LDUNet_MVA_ft',type=str)
    args = parser.parse_known_args()[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    train(args)
