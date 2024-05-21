import torch
import torch.nn as nn
from blocks import ResBlock, UpSkip


#Patch-based multi-view self-attention block
class PMSA(nn.Module):
    def __init__(self,in_channels,an,patch_size,num_heads):
        super(PMSA,self).__init__()
        self.an=an
        self.patch_size=patch_size
        self.num_heads=num_heads
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.norm=nn.LayerNorm(in_channels)
        self.qk=nn.Linear(in_channels,in_channels*2)
        self.scale = (in_channels/num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.conv=nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        
    def forward(self,inp):
        #inp: [B*A*A,C,H,W]
        N,C,H,W=inp.shape
        B=N//(self.an*self.an)
        ph=H//self.patch_size
        pw=W//self.patch_size
        inp_re = inp.reshape(N, C, ph, self.patch_size, pw, self.patch_size)
        inp_re = inp_re.permute(0,2,4,1,3,5).reshape(B,self.an*self.an,ph*pw,C,self.patch_size,self.patch_size).permute(0,2,1,3,4,5)  #[B*p*p,A*A,C,H/p,W/p]
        
        x=self.pool(inp_re.reshape(-1,C,self.patch_size,self.patch_size)) #[B*p*p*A*A,C,1,1]
        x=x.flatten(1)  #[B*p*p*A*A,C]
        x=self.norm(x)  #[B*p*p*A*A,C]
        x=x.reshape(B*ph*pw,self.an*self.an,C)  #[B*p*p,A*A,C]
        qk=self.qk(x).reshape(B*ph*pw,self.an*self.an,2,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)   #[2,B*p*p,head,A*A,C/head]
        q,k = qk.unbind(0)   #[B*p*p,head,A*A,C/head]
        att = (q @ k.transpose(-2, -1))*self.scale  #[B*p*p,head,A*A,A*A]
        att = self.softmax(att)   #[B*p*p,head,A*A,A*A]
        v=inp_re.reshape(B*ph*pw,self.an*self.an,self.num_heads,-1).permute(0,2,1,3)  #[B*p*p,head,A*A,(C/head)*(H/p)*(W/p)]
        x=(att@v).permute(0,2,1,3).reshape(B,ph,pw,self.an*self.an,C,self.patch_size,self.patch_size).permute(0,3,4,1,5,2,6).reshape(B*self.an*self.an,C,H,W)  #[B*A*A,C,H,W]
        out=inp+self.conv(x)   #[B*A*A,C,H,W]
        
        return out
    
    
class LFRefineNet(nn.Module):
    def __init__(self,in_channels,out_channels,base_channels,an,patch_size):
        super(LFRefineNet,self).__init__()
        num_channels=[base_channels, base_channels*2, base_channels*4, base_channels*8]

        self.conv1=nn.Sequential(nn.Conv2d(in_channels, num_channels[0], kernel_size=3, stride=1, padding=1),
                                 nn.GroupNorm(8,num_channels[0]),
                                 nn.LeakyReLU(inplace=True))

        self.block1=ResBlock(num_channels[0],num_channels[1],stride=2)
        self.pmsa1=PMSA(num_channels[1], an, patch_size, num_heads=4)

        self.block2=ResBlock(num_channels[1],num_channels[2],stride=2)
        self.pmsa2=PMSA(num_channels[2], an, patch_size, num_heads=8)

        self.block3=ResBlock(num_channels[2],num_channels[3],stride=2)
        self.pmsa3=PMSA(num_channels[3], an, patch_size, num_heads=16)

        self.skip1=UpSkip(num_channels[3],num_channels[2],mode='sum')
        self.block4=ResBlock(num_channels[2],num_channels[2],stride=1)
        self.pmsa4=PMSA(num_channels[2], an, patch_size, num_heads=8)
        
        self.skip2=UpSkip(num_channels[2],num_channels[1],mode='sum')
        self.block5=ResBlock(num_channels[1],num_channels[1],stride=1)
        self.pmsa5=PMSA(num_channels[1], an, patch_size, num_heads=4)

        self.skip3=UpSkip(num_channels[1],num_channels[0],mode='sum')
        self.block6=ResBlock(num_channels[0],num_channels[0],stride=1)
        
        self.conv2=nn.Conv2d(num_channels[0],out_channels,kernel_size=3,stride=1,padding=1)
                                
        
    def forward(self,inp):
        #inp: [B*A*A,6,H,W]
        map1=self.conv1(inp)  #[B*A*A,C,H,W]
        
        map2=self.block1(map1)  #[B*A*A,2C,H/2,W/2]
        map2=self.pmsa1(map2)  #[B*A*A,2C,H/2,W/2]
        
        map3=self.block2(map2)  #[B*A*A,4C,H/4,W/4]
        map3=self.pmsa2(map3)   #[B*A*A,4C,H/4,W/4]

        map4=self.block3(map3)  #[B*A*A,8C,H/8,W/8]
        map4=self.pmsa3(map4)   #[B*A*A,8C,H/8,W/8]
        
        map3=self.skip1(map4,map3)  #[B*A*A,4C,H/4,W/4]
        map3=self.block4(map3)     #[B*A*A,4C,H/4,W/4]
        map3=self.pmsa4(map3)      #[B*A*A,4C,H/4,W/4]
        
        map2=self.skip2(map3,map2)  #[B*A*A,2C,H/2,W/2]
        map2=self.block5(map2)    #[B*A*A,2C,H/2,W/2]
        map2=self.pmsa5(map2)     #[B*A*A,2C,H/2,W/2]
        
        map1=self.skip3(map2,map1)  #[B*A*A,C,H,W]
        map1=self.block6(map1)      #[B*A*A,C,H,W]
        
        out=self.conv2(map1)+inp[:,:3,:,:]   #[B*A*A,3,H,W]
        out=torch.clamp(out,-1,1) 
        
        return out
        
    
