import torch
import torch.nn as nn
import math
from blocks import ResBlock


def TimeEmbedding(t,dim):
    assert len(t.shape) == 1 #[B]

    dim_half = dim // 2
    c = math.log(10000) / (dim_half - 1)
    c = torch.exp(torch.arange(dim_half, dtype=torch.float32) * -c)
    c = c.to(device=t.device)
    emb = t.float()[:, None] * c[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  #[B,D]
    if dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


#Residual block with time embedding
class ResTimeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride,t_channels):
       super(ResTimeBlock,self).__init__() 
       self.conv1=nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                 nn.GroupNorm(num_groups=8, num_channels=out_channels),
                                 nn.LeakyReLU(inplace=True))
       self.t_proj = nn.Linear(t_channels,out_channels)
       self.conv2=nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                 nn.GroupNorm(num_groups=8, num_channels=out_channels))
       if in_channels != out_channels:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
                                            nn.GroupNorm(num_groups=8, num_channels=out_channels))
       else:
            self.shortcut=None
       self.relu=nn.LeakyReLU(inplace=True)
       self.stride=stride
       
    def forward(self,x,t):
        residual=self.conv1(x)
        residual=residual+self.t_proj(t)[:,:,None,None]
        residual=self.conv2(residual)
        if self.shortcut:
            x=self.shortcut(x)
        if self.stride!=1:
            x=nn.functional.interpolate(x,scale_factor=0.5,mode='bilinear')
        out=self.relu(x+residual)
        return out
    

#Condition encoder
class ConEncoder(nn.Module):
    def __init__(self,in_channels,num_channels):
        super(ConEncoder,self).__init__()
        
        self.conv=nn.Sequential(nn.Conv2d(in_channels, num_channels[0], kernel_size=3, stride=2, padding=1),
                                 nn.GroupNorm(8,num_channels[0]),
                                 nn.LeakyReLU(inplace=True))
        
        self.block=ResBlock(num_channels[0], num_channels[0],stride=2)
        
        self.block1=ResBlock(num_channels[0], num_channels[0],stride=2)
        
        self.block2=ResBlock(num_channels[0], num_channels[1],stride=2)
        
        self.block3=ResBlock(num_channels[1], num_channels[2],stride=2)
        
        self.block4=ResBlock(num_channels[2], num_channels[3],stride=2)
    
    def forward(self,inp):
        feat1=self.conv(inp)      #[B*A*A,C,H/2,W/2]
        feat2=self.block(feat1)   #[B*A*A,C,H/4,W/4]
        
        map1=self.block1(feat2)  #[B*A*A,C,H/8,W/8]
        map2=self.block2(map1)   #[B*A*A,2C,H/16,W/16]
        map3=self.block3(map2)   #[B*A*A,4C,H/32,W/32]
        map4=self.block4(map3)   #[B*A*A,8C,H/64,W/64]
        
        return map1,map2,map3,map4


#Cross-attention
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, pool_size):
        super(CrossAttention,self).__init__()
        self.num_heads=num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        
        self.pool=nn.AdaptiveAvgPool2d(pool_size)
        self.norm1 = nn.LayerNorm(dim)
        self.proj_diff=nn.Linear(dim, dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim,dim)
        
        self.norm3=nn.LayerNorm(dim)
        self.proj_con=nn.Linear(dim, dim)
           
        self.norm4=nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, 2*dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self,feat_diff,feat_con):
        #feat_diff: [B,c,h,w] feat_con: [B,c,h,w]
        B,c,h,w=feat_diff.shape
        anchor_diff=self.pool(feat_diff)  #[B,c,hs,ws]
        anchor_diff=self.norm1(anchor_diff.flatten(2).transpose(1,2)) #[B,hs*ws,c]
        anchor_diff=self.proj_diff(anchor_diff).reshape(B,-1,self.num_heads,c//self.num_heads).permute(0,2,1,3)  #[B,head,hs*ws,c/head] 
        
        x_diff=self.norm2(feat_diff.flatten(2).transpose(1,2)) #[B,h*w,c]
        q=self.q(x_diff).reshape(B,-1,self.num_heads,c//self.num_heads).permute(0,2,1,3)  #[B,head,h*w,c/head]
        
        anchor_con=self.pool(feat_con)  #[B,c,hs,ws]
        anchor_con=self.norm3(anchor_con.flatten(2).transpose(1,2)) #[B,hs*ws,c]
        anchor_con=self.proj_con(anchor_con).reshape(B,-1,self.num_heads,c//self.num_heads).permute(0,2,1,3)  #[B,head,hs*ws,c/head]
        
        x_con=self.norm4(feat_con.flatten(2).transpose(1,2)) #[B,h*w,c]
        kv=self.kv(x_con).reshape(B,-1,2,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)  #[2,B,head,h*w,c/head]
        k,v=kv[0],kv[1]  #[B,head,h*w,c/head]
        
        att1 = self.softmax((anchor_diff @ k.transpose(-2,-1))*self.scale)  #[B,head,hs*ws,h*w] 
        out1 = att1 @ v    #[B,head,hs*ws,c/head]
        
        att2 = self.softmax((q @ anchor_con.transpose(-2,-1))*self.scale) #[B,head,h*w,hs*ws]
        out2 = (att2 @ out1).permute(0,2,1,3).reshape(B,-1,c)  #[B,h*w,c]
        
        out2 = self.proj(out2).reshape(B,h,w,c).permute(0,3,1,2)   #[B,c,h,w]
        
        out=feat_diff+out2   #[B,c,h,w]
        
        return out
    
    
class UpSkip(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpSkip, self).__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self,feat,feat_diff,feat_con):
        feat=self.upsample(feat)
        feat=self.conv(feat)
        out=feat+feat_diff+feat_con
        return out


class LDUNet(nn.Module):
    def __init__(self,in_channels,con_channels,base_channels,t1_channels,t2_channels,pool_sizes):
        super(LDUNet,self).__init__()
        num_channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        self.con_encoder=ConEncoder(con_channels, num_channels)

        self.t1_channels=t1_channels
        self.t_emb=nn.Sequential(nn.Linear(t1_channels,t2_channels), nn.LeakyReLU(inplace=True),
                                nn.Linear(t2_channels,t2_channels), nn.LeakyReLU(inplace=True))
        
        self.conv1=nn.Sequential(nn.Conv2d(in_channels, num_channels[0], kernel_size=3, stride=1, padding=1),
                                 nn.GroupNorm(8,num_channels[0]),
                                 nn.LeakyReLU(inplace=True))
        self.cross_att1=CrossAttention(num_channels[0], num_heads=8, pool_size=pool_sizes[0])
        
        self.down_block1=ResTimeBlock(num_channels[0],num_channels[1],stride=2,t_channels=t2_channels)
        self.cross_att2=CrossAttention(num_channels[1], num_heads=8, pool_size=pool_sizes[1])
        
        self.down_block2=ResTimeBlock(num_channels[1],num_channels[2],stride=2,t_channels=t2_channels)
        self.cross_att3=CrossAttention(num_channels[2], num_heads=8, pool_size=pool_sizes[2])
        
        self.down_block3=ResTimeBlock(num_channels[2],num_channels[3],stride=2,t_channels=t2_channels)
        self.cross_att4=CrossAttention(num_channels[3], num_heads=8, pool_size=pool_sizes[3])
        
        self.up1=UpSkip(num_channels[3],num_channels[2])
        self.up_block1=ResTimeBlock(num_channels[2], num_channels[2],stride=1,t_channels=t2_channels)
        
        self.up2=UpSkip(num_channels[2],num_channels[1])
        self.up_block2=ResTimeBlock(num_channels[1], num_channels[1],stride=1,t_channels=t2_channels)
        
        self.up3=UpSkip(num_channels[1],num_channels[0])
        self.up_block3=ResTimeBlock(num_channels[0], num_channels[0],stride=1,t_channels=t2_channels)
        
        self.conv_final=nn.Conv2d(num_channels[0],in_channels,kernel_size=3,stride=1,padding=1)
                                      
        
    def forward(self,zt,x_con,t):
        #xt:[B,4,h,w], x_con: [B,3,H,W] t:[B]
        map1_con,map2_con,map3_con,map4_con=self.con_encoder(x_con)
        
        temb=TimeEmbedding(t,self.t1_channels) #[B,D]
        temb=self.t_emb(temb)  #[B,D]

        map1=self.conv1(zt)  #[B,c,h,w]
        map1=self.cross_att1(map1,map1_con)  #[B,c,h,w]
        
        map2=self.down_block1(map1,temb) #[B,2c,h/2,w/2]
        map2=self.cross_att2(map2,map2_con)  #[B,2c,h/2,w/2]
        
        map3=self.down_block2(map2,temb)  #[B,4c,h/4,w/4]
        map3=self.cross_att3(map3,map3_con)   #[B,4c,h/4,w/4]
        
        map4=self.down_block3(map3,temb) #[B,8c,h/8,w/8]
        map4=self.cross_att4(map4,map4_con)  #[B,8c,h/8,w/8]
        
        map3=self.up1(map4,map3,map3_con) #[B,4c,h/4,w/4]
        map3=self.up_block1(map3,temb) #[B,4c,h/4,w/4]
        
        map2=self.up2(map3,map2,map2_con) #[B,2c,h/2,w/2]
        map2=self.up_block2(map2,temb) #[B,2c,h/2,w/2]
        
        map1=self.up3(map2,map1,map1_con) #[B,c,h,w]
        map1=self.up_block3(map1,temb) #[B,c,h,w]
        
        out=self.conv_final(map1) #[B,4,h,w]
        
        return out
 