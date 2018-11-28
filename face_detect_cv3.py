import cv2
import predict_image

# Get user supplied values
imagePath = 'D:/Communications/My_Graduation_project/Facedet_Final_version/Behidy87.jpg'
cascPath = "haarcascade_frontalface_default.xml"
t = 2 #thickness of the boundary box
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    crop_image = image[y+t:y+h-t, x+t:x+w-t]
    test_im = predict_image.SqueezeDet_Model()
    test_im.layers()
    pre = test_im.predict_neural_network(crop_image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if int(pre) == 0:
        cv2.putText(image, "Kassify!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
    elif int(pre) == 1:
        cv2.putText(image, "Salama!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
    elif int(pre) == 2:
        cv2.putText(image, "Ali!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
    elif int(pre) == 3:
        cv2.putText(image, "Essam!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
    elif int(pre) == 4:
        cv2.putText(image, "Mustafa!", (x-2*t,y-2*t), font, 0.8, (0, 255, 0))
        
        

cv2.imshow("Faces found", image)
cv2.waitKey(0)
