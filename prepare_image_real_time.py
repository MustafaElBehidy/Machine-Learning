# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 04:49:48 2017

@author: MAHMOUD
"""
import numpy as np
import cv2

def preparing_image(image,image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image , (image_size , image_size ) )
    image = image.flatten()
    return image
    
def load_image(image,image_size,Conv=True):
    image = preparing_image(image,image_size)
    if Conv:
        #image = np.reshape(image , (image_size,image_size,1) )
        image = np.reshape(image,(-1,image_size,image_size,3))
    return image