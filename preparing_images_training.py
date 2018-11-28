import numpy as np
import cv2
import os
from sklearn import preprocessing

def traverse_dir(path,image_size,image_channels,pre_processing):
    
    images = []
    images = np.array(images)
    i = 0
    #print (len(os.listdir(path)))
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        
        if file_or_dir.endswith('.jpg'):
                image = cv2.imread(abs_path)
                
                if image_channels == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                elif image_channels == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                image = cv2.resize(image , (image_size , image_size ) )
                image = np.array(image)
                
                if pre_processing:
                    image = preprocessing.scale(image)
                    image /= 255
                  
                image = image.flatten()
                if i != 0:
                    images = np.vstack((images, image))
                else:
                    images = image
                    i = 1
                
    
    return images
    
def load_images(path,image_size,image_channels,Conv=False,pre_processing=False):
    
    images = traverse_dir(path,image_size,image_channels,pre_processing)
    if Conv:
        images = np.reshape(images,(-1 ,image_size,image_size,image_channels))
    return images

class Batch(object):
  def __init__(self, X, y, batch_size):
    self.batch_size = batch_size
    self.X = X
    self.y = y
    self.size = X.shape[0]
    self.start = 0
  def getBatch_random(self):
    indices = np.random.choice(range(self.size), self.batch_size,False)
    return self.X[indices, :], self.y[indices]
     
  def getBatch_series(self):
    check_availability = int(self.batch_size/self.size)
    self.end = self.start + self.batch_size
    
    if(check_availability == 0):
        if (self.end <= self.size):
            X_temp = self.X[self.start : self.end , :]
            y_temp = self.y[self.start : self.end ]
            self.start = self.start + self.batch_size
        else:
            s = np.arange (self.size)
            np.random.shuffle(s)
            self.X = self.X[s,:]
            self.y = self.y[s]
            self.start = 0
            self.end = self.start + self.batch_size
            X_temp = self.X[self.start : self.end , :]
            y_temp = self.y[self.start : self.end ]
            self.start = self.start + self.batch_size
          
        X_temp = np.reshape(X_temp,(-1,224,224,3))
        return X_temp , y_temp
              
    else:
        self.X = np.reshape(self.X,(-1,224,224,3))
        return self.X, self.y
            
