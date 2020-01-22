import cv2
import numpy as np
import os

def create_face_data():
	front_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	list_img = os.listdir("Face-PreProcessing")
	i = 1
	for fn in list_img:
		img = cv2.imread("Face-PreProcessing/" + fn)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = front_cascade.detectMultiScale(gray, 1.3, 6)
		for a,b,c,d in faces:
			print(a,b,c,d)
			faceimg = img[b:b+d, a:a+c]
			ff = cv2.resize(faceimg,(24,24))
			cv2.imwrite("FaceData/" + str(i) + '.jpg',ff)
			i += 1

def create_nonface_data():
	list_img = os.listdir("NonFace-PreProcessing")
	i = 1
	for fn in list_img:
		img = cv2.imread("NonFace-PreProcessing/" + fn, 0)
		ff = cv2.resize(img,(24,24))
		cv2.imwrite("NonFaceData/" + str(i) + '.jpg',ff)
		i += 1
		
# create_face_data()
# create_nonface_data()


