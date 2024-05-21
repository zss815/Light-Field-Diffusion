import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F

from utils.losses import SSIM


class RestoreLoss(nn.Module):
    def __init__(self):
        super(RestoreLoss,self).__init__()
        self.l1_loss=nn.L1Loss() 
        self.ssim=SSIM(window_size=11,size_average=True)
        vgg = vgg16(pretrained=True)
        self.vgg_layers = nn.Sequential(*list(vgg.features)[:31]).cuda().eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
    def forward(self,pred,gt):
        l1_loss=self.l1_loss(pred,gt)
        ssim_loss=1-self.ssim(pred,gt)
        vgg_loss=self.l1_loss(self.vgg_layers(pred), self.vgg_layers(gt))
        
        total_loss=1*l1_loss+1*ssim_loss+1*vgg_loss
        print('l1_loss: {:.4f}, ssim_loss: {:.4f}, vgg_loss: {:.4f}'
              .format(l1_loss,ssim_loss,vgg_loss))
        
        return total_loss


