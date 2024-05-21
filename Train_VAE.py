import os
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset

from Dataset import VAETrainData, VAETestData
from Loss import RestoreLoss
from utils.metrics import PSNR, SSIM
from networks.VAE import VAE


def adjust_learning_rate(optimizer):
    lr=optimizer.param_groups[0]['lr']
    lr=lr*0.9
    if lr>1e-4:
     	optimizer.param_groups[0]['lr']=lr
        
        
def train(args):
    lr_epoch=50
    epoch_dict,psnr_dict,ssim_dict={},{},{}
    for i in range(1,args.save_num+1):
        epoch_dict[str(i)]=0
        psnr_dict[str(i)]=0
        ssim_dict[str(i)]=0
    best_psnr=0
    best_ssim=0
    best_epoch=0
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root,exist_ok=True)
            
    train_set1=VAETrainData('/mnt/disk/LFLL/LFReal/Train')
    train_set2=VAETrainData('/mnt/disk/LFLL/L3F/GT')
    train_set=ConcatDataset([train_set1,train_set2])
    
    test_set=VAETestData('/mnt/disk/LFLL/LFReal/Test')
    test_num=len(test_set)

    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_set,batch_size=args.batch_size,shuffle=False)
    
    model = VAE(in_channels=3,z_channels=4,hid_channels=[64,128,256,512],strides=[1,2,2,2])
    print('model parameters: {}'.format(sum(param.numel() for param in model.parameters())))
    
    criterion=RestoreLoss()
    
    model.cuda() 
    criterion.cuda()
    
    if args.pre_train:
        model.load_state_dict(torch.load(args.model_path))
        
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr_init)
    
    for epoch in range(args.max_epoch):
        model.train()
        if epoch % lr_epoch==0 and epoch!=0:
            adjust_learning_rate(optimizer)
        
        for idx,inp in enumerate(train_loader):
            inp = inp.cuda()
            optimizer.zero_grad()
            
            pred,kl_loss=model(inp)
            
            img_loss=criterion(pred,inp)
            loss=img_loss+1e-6*kl_loss
            
            loss.backward()
            optimizer.step()
            print('Epoch: %i, step: %i, kl_loss: %f' %(epoch,idx,kl_loss.item()))
            print('')
            
        model.eval()
        with torch.no_grad():
            total_psnr=0
            total_ssim=0
            
            for inp in test_loader:
                inp=inp.cuda()
                pred,_=model(inp)
                pred=pred.cpu().numpy()
                inp=inp.cpu().numpy()
                pred=(pred+1)/2
                inp=(inp+1)/2
                B=inp.shape[0]
                for i in range(B):
                    p=pred[i]
                    g=inp[i]
                    psnr=PSNR(g,p,data_range=1)
                    ssim=SSIM(g,p,data_range=1)
                    total_psnr+=psnr
                    total_ssim+=ssim
                        
            ave_psnr=total_psnr/test_num
            ave_ssim=total_ssim/test_num
            print('Epoch {}, average PSNR {}, SSIM {}'.format(epoch,ave_psnr,ave_ssim))
            print('')
    
        #save models            
        if epoch<args.save_num:
            torch.save(model.state_dict(),os.path.join(args.save_root,'model%s.pth'%str(epoch+1)))
            psnr_dict[str(epoch+1)]=ave_psnr
            ssim_dict[str(epoch+1)]=ave_ssim
            epoch_dict[str(epoch+1)]=epoch
        else:
            if ave_psnr>min(psnr_dict.values()):
                torch.save(model.state_dict(),os.path.join(args.save_root,'model%s.pth'%(min(psnr_dict,key=lambda x: psnr_dict[x]))))
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
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr_init', default=1e-3, type=float)
    parser.add_argument('--max_epoch',default=1000,type=int)
    parser.add_argument('--save_num', default=10, type=int, help='number of saved models')
    parser.add_argument('--save_root',default='/mnt/disk/save_models/Diffusion/Lowlight/VAE',type=str)
    parser.add_argument('--pre_train',default=False,type=bool)
    parser.add_argument('--model_path',default='/mnt/disk/save_models/Diffusion/Lowlight/VAE/model1.pth',type=str)
    args = parser.parse_known_args()[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    train(args)
    
