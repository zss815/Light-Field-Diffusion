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

  
