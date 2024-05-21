import torch
import torch.nn as nn
from blocks import ResBlock



class DiagonalGaussianDistribution(nn.Module):
    def __init__(self):
        super(DiagonalGaussianDistribution,self).__init__()

    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        B = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)/B

        return z, kl 
    
    
class VAE(nn.Module):
    def __init__(self,in_channels,z_channels,hid_channels=[64,128,256,512],strides=[1,2,2,2]):
        super(VAE,self).__init__()
        
        depth=len(hid_channels)
        self.conv=nn.Sequential(nn.Conv2d(in_channels, hid_channels[0], kernel_size=3, stride=strides[0], padding=1),
                                 nn.GroupNorm(8,hid_channels[0]),
                                 nn.LeakyReLU(inplace=True))
        self.en_blocks=nn.ModuleList([ResBlock(hid_channels[i-1],hid_channels[i],stride=strides[i]) for i in range(1,depth)])
        
        self.conv_z1=nn.Sequential(nn.Conv2d(hid_channels[-1], z_channels*2, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(z_channels*2, z_channels*2, kernel_size=1, stride=1))
        
        self.sample=DiagonalGaussianDistribution()
        
        self.conv_z2=nn.Sequential(nn.Conv2d(z_channels, hid_channels[-1], kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(8,hid_channels[-1]),
                                   nn.LeakyReLU(inplace=True))
        
        de_blocks=[]
        for i in range(depth-1,0,-1):
            if strides[i]==2:
                de_blocks.append(nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear'),
                                               ResBlock(hid_channels[i], hid_channels[i-1],stride=1)))
            else:
                de_blocks.append(ResBlock(hid_channels[i], hid_channels[i-1],stride=1))
        self.de_blocks=nn.ModuleList(de_blocks)
        
        self.conv_out=nn.Conv2d(hid_channels[0], in_channels, kernel_size=3, stride=1, padding=1)
        
    def encoder(self,inp):
        x=self.conv(inp)
        for block in self.en_blocks:
            x=block(x)
        z=self.conv_z1(x)
        z,_=self.sample(z)

        return z
    
    def decoder(self,z):
        x=self.conv_z2(z)
        for block in self.de_blocks:
            x=block(x)
        out=self.conv_out(x)
        out=torch.clamp(out,-1,1)

        return out

    def forward(self,inp):
        x=self.conv(inp)
        for block in self.en_blocks:
            x=block(x)
        z=self.conv_z1(x)
        z,kl_loss=self.sample(z)
        
        x=self.conv_z2(z)
        for block in self.de_blocks:
            x=block(x)
        out=self.conv_out(x)
        out=torch.clamp(out,-1,1)
        
        return out,kl_loss
    
        
                                   
        