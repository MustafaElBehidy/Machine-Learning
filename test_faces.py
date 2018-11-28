# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 04:48:39 2017

@author: MAHMOUD
"""

import cv2
import logging as log
import datetime as dt
from time import sleep
import predict_image

cascPath1 = "haarcascade_frontalface_default.xml"
cascPath2 = "haarcascade_profileface.xml"


faceCascade = cv2.CascadeClassifier(cascPath1)
alt_faceCascade = cv2.CascadeClassifier(cascPath2)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
t = 2 #thickness of the boundary box
counter = 0

# 1 second in about 11 loop
num_counter = 11 * 5 #sav image every 3 seconds

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if (counter % num_counter) == 0:
            crop_image = frame[y+t:y+h-t, x+t:x+w-t]
            test_im = predict_image.SqueezeDet_Model()
            test_im.layers()
            pre = test_im.predict_neural_network(crop_image) 
            if int(pre) == 0:
                cv2.putText(frame, "Kassify!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
            elif int(pre) == 1:
                cv2.putText(frame, "Salama!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
            elif int(pre) == 2:
                cv2.putText(frame, "Ali!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
            elif int(pre) == 3:
                cv2.putText(frame, "Essam!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
            elif int(pre) == 4:
                cv2.putText(frame, "Mustafa!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
            
        
    counter = counter + 1

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
    
    # Display the resulting frame
    #cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
