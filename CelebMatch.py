import cv2
import numpy as np

#Importing Haar Cascade Classifier to recognize face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Importing the images and converting to Grayscale for processing
img1 = cv2.imread('images/me.png',1)
img2 = cv2.imread('images/celeb.jpg',1)
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#Extracting Region of Interest using Haar Cascade
face1 = face_cascade.detectMultiScale(img1_gray,1.3,5)
face2 = face_cascade.detectMultiScale(img2_gray,1.3,5)

for (x,y,w,h) in face1:
	roi_gray_1 = img1_gray[y:y+h, x:x+w]
	
for (x,y,w,h) in face2:
	roi_gray_2 = img2_gray[y:y+h, x:x+w]


#Finding and matching Keypoints	
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(roi_gray_1, None)
kp2, des2 = sift.detectAndCompute(roi_gray_2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

#Applying ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

score = len(good)

msg = "Your match score with the celebrity is " + str(score) + "!"
font=cv2.FONT_HERSHEY_SIMPLEX

#Putting Text on the Output Image
img_out = np.concatenate((img1,img2),axis=1)
cv2.putText(img_out,msg,(20,450),font,0.85,(255,255,0),2,cv2.LINE_AA)

#Showing the match score on the output image
cv2.imshow('Out',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
