# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:38:11 2017

@author: MAHMOUD
"""
import predict_image
import cv2
import time

path = 'D:/Communications/My_Graduation_project/Facedet_Final_version/Essam4.jpg'
s = time.time()
image = cv2.imread(path)
test_im = predict_image.SqueezeDet_Model()
test_im.layers()
pre = test_im.predict_neural_network(image)
