import numpy as np 
import cv2
import pickle
from IntegralImage import to_integral_image
def subimg(h,w , minsz, maxsi, stride):
	l = []
	for size in range(minsz,maxsi,2):
		for i in range(0,h-size,int(size/3)):
			for j in range(0,w-size,int(size/3)):
				l.append([i,j,size,size])
	return l

path = "img_51.jpg"
img = cv2.imread(path, 0)

h = img.shape[0]
w = img.shape[1]

subwindw = subimg(h,w,50,150,5)
print(len(subwindw))
model = pickle.load(open("abc", 'rb'))
l = []
for y,x,h,w in subwindw:
	v = model.predict(to_integral_image(cv2.resize(img[y:y+h, x:x+w],(24,24))))
	if v:
		# if any((y_ > y and y_ < y + h and x_ > x and x_ < x + w) or (y_ + h_ > y and y_ + h_ < y + h and x_ + w_ > x and x_ + w_ < x + w) for y_,x_,h_,w_ in l):
		# 	continue
		l.append([y,x,h,w])
		print(y,x,h,w)
		# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.imshow('img',img[y:y+h, x:x+w])
		cv2.waitKey(0)
		cv2.destroyAllWindows()

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()