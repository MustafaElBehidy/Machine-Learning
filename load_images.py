# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 03:35:48 2017

@author: MAHMOUD
"""
import numpy as np
import preparing_images_training
import scipy.io as sio

path1 = "D:/4th year 2nd term/New folder/dataSet/dataSet/Ahmed_Essam"
path2 = "D:/4th year 2nd term/New folder/dataSet/dataSet/Mahmoud_Kassify"
path3 = "D:/4th year 2nd term/New folder/dataSet/dataSet/Mahmoud_Salama"
path4 = "D:/4th year 2nd term/New folder/dataSet/dataSet/Mustafa_Behidy"
path5 = "D:/4th year 2nd term/New folder/dataSet/dataSet/Muhammad_Ali"


image_size = 224
image_channels = 3

#load_images(path,image_size,image_channels,Conv=False,pre_processing=False)

#Conv means you want this images for convolutional neural network or not
#pre_processing means if you want to make preprocessing for the data or not
    #note preprocessing for gray or binary images 
  
#im = np.array(preparing_images_training.load_images(path,image_size,image_channels,False,False))
#sio.io.savemat('Kassify_photos.mat',{'Kassify':im})
#o = np.reshape(im[4],(image_size,image_size))
#plt.imshow(o, cmap='gray')

im_1 = np.array(preparing_images_training.load_images(path1,image_size,image_channels,False,False))
im_2 = np.array(preparing_images_training.load_images(path2,image_size,image_channels,False,False))
im_3 = np.array(preparing_images_training.load_images(path3,image_size,image_channels,False,False))
im_4 = np.array(preparing_images_training.load_images(path4,image_size,image_channels,False,False))
im_5 = np.array(preparing_images_training.load_images(path5,image_size,image_channels,False,False))

s = np.arange(im_1.shape[0])
np.random.shuffle(s)
Essam_images = im_1[s]
Essam_labels = 3*np.ones((im_1.shape[0]), dtype=np.int)

s = np.arange(im_2.shape[0])
np.random.shuffle(s)
Kassify_images = im_2[s]
Kassify_labels = 0*np.ones((im_2.shape[0]), dtype=np.int)

s = np.arange(im_3.shape[0])
np.random.shuffle(s)
Salama_images = im_3[s]
Salama_labels = 1*np.ones((im_3.shape[0]), dtype=np.int)

s = np.arange(im_4.shape[0])
np.random.shuffle(s)
Behidy_images = im_4[s]
Behidy_labels = 4*np.ones((im_4.shape[0]), dtype=np.int)

s = np.random.choice(im_5.shape[0], 500 , False)
Ali_images = im_5[s]
Ali_labels = 2*np.ones((500), dtype=np.int)

print(Ali_images.shape , Ali_labels.shape)
print(Essam_images.shape , Essam_labels.shape)
print(Salama_images.shape , Salama_labels.shape)
print(Kassify_images.shape , Kassify_labels.shape)
print(Behidy_images.shape , Behidy_labels.shape)

images = np.vstack((Essam_images, Kassify_images))
images = np.vstack((images, Salama_images))
images = np.vstack((images, Behidy_images))
images = np.vstack((images, Ali_images))

labels = np.hstack((Essam_labels, Kassify_labels))
labels = np.hstack((labels, Salama_labels))
labels = np.hstack((labels, Behidy_labels))
labels = np.hstack((labels, Ali_labels))

s = np.arange(images.shape[0])
np.random.shuffle(s)

faces_data_colors_224 = {}
faces_data_colors_224['images'] = images[s]
faces_data_colors_224['labels'] = labels[s]

print(faces_data_colors_224['images'].shape , faces_data_colors_224['labels'].shape)
sio.savemat('faces_data_colors_224',faces_data_colors_224)
