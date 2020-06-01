import cv2
import numpy as np
import urllib

url='http://192.168.43.1:8080/shot.jpg'
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#cam=cv2.VideoCapture(0); # code for laptop camera

rec=cv2.createLBPHFaceRecognizer();
#rec=cv2.face.LBPHFaceRecognizer_create();
rec.load("recognizer\\trainingData.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
#font=cv2.FONT_HERSHEY_SIMPLEX
while (True):
     imgResp=urllib.urlopen(url)
     imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
     img=cv2.imdecode(imgNp,-1)
     
     #ret,img=cam.read(); # code for laptop camera
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     faces=faceDetect.detectMultiScale(gray,1.3,5);
     for(x,y,w,h) in faces:
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
         id,conf=rec.predict(gray[y:y+h,x:x+w])
         if(id==1):
             id="1"
         cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
     cv2.imshow("Faces",img);
     if(cv2.waitKey(1)==ord('q')):
         break;
#cam.release()
cv2.destroyAllWindows()
 
