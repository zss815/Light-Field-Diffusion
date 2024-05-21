import numpy as np
import cv2
  
def illum_adjust(img,v_scale):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    #Brightness
    if v_scale:
        v=img_hsv[...,2]
        v=v.astype(np.float64)
        v=v*v_scale
        v[v>255]=255
        v=v.astype(np.uint8)
        img_hsv[...,2]=v
    img=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
    return img

#Low-light LF synthesis
def LowLF_syn(gt_mp,v_scale,k):
    low_mp=illum_adjust(gt_mp,v_scale)
    raw_mp = RGB2Bayer(low_mp).astype(np.float64)

    log_read_mean=0.6*np.log(k)+0.6
    log_read=np.random.uniform(log_read_mean-0.1,log_read_mean+0.1)
    read_scale=np.exp(log_read)

    log_row_mean=0.3*np.log(k)
    log_row=np.random.uniform(log_row_mean-0.1,log_row_mean+0.1)
    row_scale=np.exp(log_row)

    raw_noise_mp = Noise_syn(raw_mp,k,read_scale,row_scale,q=1)
    inp_mp = cv2.cvtColor(raw_noise_mp, cv2.COLOR_BAYER_GB2RGB)
    
    return inp_mp

  
