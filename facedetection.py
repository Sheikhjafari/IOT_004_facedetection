import numpy as np
import cv2

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(1)

while(True):
    ret, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.8,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w+10,y+h+10),(0,255,255),5)
    cv2.imshow("Face",img);
    if (cv2.waitKey(1)==ord('q')):
        break;
cam.release();
cv2.destroyAllWindows();

    

