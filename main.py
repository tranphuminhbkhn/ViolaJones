from __future__ import division
import numpy as np
from IntegralImage import to_integral_image
from ViolaJones import ViolaJonesModel
from Feature import Feature
import os
import cv2
import pickle
import random


Fp = "FaceData"
nFp = "NonFaceData"

list_faceimg = os.listdir(Fp)
for i in range(len(list_faceimg)):
	fn = list_faceimg[i]
	img = cv2.imread(Fp + "/" + fn, 0)
	if i == 0:
		FaceXtrain = np.array([to_integral_image(img)])
		continue
	FaceXtrain = np.concatenate((FaceXtrain, [to_integral_image(img)]))


list_faceimg = os.listdir(nFp)
for i in range(len(list_faceimg)):
	fn = list_faceimg[i]
	# print(fn)
	img = cv2.imread(nFp + "/" + fn, 0)
	if i == 0:
		NonFaceXtrain = np.array([to_integral_image(img)])
		continue
	NonFaceXtrain = np.concatenate((NonFaceXtrain, [to_integral_image(img)]))

print(FaceXtrain.shape)
print(NonFaceXtrain.shape)



model = ViolaJonesModel()
model.fit(FaceXtrain, NonFaceXtrain ,20, 0.4, 0.99)
model.save_model("abc")

model = pickle.load(open("abc", 'rb'))

s = 0
list_faceimg = os.listdir("FaceData")
for i in range(len(list_faceimg)):
	fn = list_faceimg[i]
	img = cv2.imread("FaceData" + "/" + fn, 0)
	# print(model.model[0][0].get_vote(to_integral_image(img)))
	p = model.predict(to_integral_image(img))
	s += p


print("Face Acc :", s/len(list_faceimg))

# model = pickle.load(open("abc", 'rb'))
s = 0
list_faceimg = os.listdir("NonFaceData")
for i in range(len(list_faceimg)):
	fn = list_faceimg[i]
	img = cv2.imread("NonFaceData" + "/" + fn, 0)
	img = cv2.resize(img,(24,24))
	p = model.predict(to_integral_image(img))
	s += (1-p)


print("NonFace Acc :", s/len(list_faceimg))