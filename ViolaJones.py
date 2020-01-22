import numpy as np
from Feature import Feature, FeatureTypes
import pickle
import math
import cv2
from IntegralImage import to_integral_image
import random
class ViolaJonesModel():
	def __init__(self):
		self.model = []
	def create_features(self, img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
	    features = []
	    print('Creating feature ...')
	    for feature in FeatureTypes:
	        feature_start_width = max(min_feature_width, feature[0])
	        for feature_width in range(feature_start_width, max_feature_width, feature[0]):
	            feature_start_height = max(min_feature_height, feature[1])
	            for feature_height in range(feature_start_height, max_feature_height, feature[1]):
	                for x in range(img_width - feature_width):
	                    for y in range(img_height - feature_height):
	                        features.append(Feature(feature, (x, y), feature_width, feature_height, 0, 1))
	                        features.append(Feature(feature, (x, y), feature_width, feature_height, 0, -1))
	    print('..done. ' + str(len(features)) + ' features created.')
	    return features



	def fit(self, FaceXtrain, NonFaceXtrain, num_of_layer, fp,tp):
		img_height, img_width = FaceXtrain[0].shape
		max_feature_height = img_height
		max_feature_width = img_width
		num_of_Facetest = 400
		num_of_Facetrain = 400
		num_of_nonFacetest = 400
		num_of_nonFacetrain = 200
		# features = self.create_features(img_height, img_width, 0, max_feature_width, 0, max_feature_height)
		features = self.create_features(img_height, img_width, 6, 10, 6, 10)
		num_of_feature = len(features)
		X = []
		Y = []
		for i in random.sample(range(len(FaceXtrain)),num_of_Facetrain):
			X.append(FaceXtrain[i])
			Y.append(1)
		for i in random.sample(range(len(NonFaceXtrain)), num_of_nonFacetrain):
			X.append(NonFaceXtrain[i])
			Y.append(0)
		X = np.array(X)
		Y = np.array(Y)

		for _ in range(num_of_layer):
			num_of_img = X.shape[0]
			print(num_of_img, np.sum(Y))
			img_weight = np.ones(num_of_img)

			layer = []
			# for i in range(num_of_img):
			# 		if Y[i] == 1:
			# 			img_weight[i] = num_of_img/ss
			false_positive = 1
			true_positive = 0
			ff = 1
			while false_positive > fp or true_positive < tp:
				print("Running in layer", _ , ", feature", ff , "...")
				ff += 1
				img_weight = img_weight/ sum(img_weight)
				score = np.zeros(num_of_feature)
				for i in range(num_of_feature):
					for j in range(num_of_img):
						v = features[i].get_vote(X[j])
						if v == Y[j]:
							score[i] += img_weight[j]


				best_idx = np.argmax(score)
				best_feature = features[best_idx]
				# print(best_idx, score[best_idx])

				best_error = 1 - score[best_idx] + 1e-9
				feature_weight = 0.5 * np.log((1 - best_error) / best_error)
				best_feature.weight = feature_weight
				layer.append(best_feature)

				for i in range(num_of_img):
					if best_feature.get_vote(X[i]) != Y[i]:
						img_weight[i] *= 2
				aa = 0
				bb = 0
				for i in random.sample(range(len(FaceXtrain)),num_of_Facetest):
					v = self.get_layer_vote(layer, FaceXtrain[i])
					if v == 1:
						aa += 1
				for i in random.sample(range(len(NonFaceXtrain)), num_of_nonFacetest):
					v = self.get_layer_vote(layer, NonFaceXtrain[i])
					if v == 1:
						bb += 1
				print(aa,bb)
				true_positive = aa/num_of_Facetest
				false_positive = bb/num_of_nonFacetest
				print(best_idx, false_positive, true_positive)
			self.model.append(layer)
			Xn = []
			Yn = []
			Ai = 0
			Bi = 0
			for i in range(num_of_img):
				v = self.predict(X[i])
				if Y[i] == 1 and v == 1:
					Xn.append(X[i])
					Yn.append(1)
					Ai += 1
					# print('A')
				if Y[i] == 0 and v == 1:
					Xn.append(X[i])
					Yn.append(0)
					Bi += 1
					# print('B')
			for i in range(Bi, int(Ai/2)):
				Xn.append(NonFaceXtrain[random.randint(1,len(NonFaceXtrain))])
				Yn.append(0)

			X = np.array(Xn)
			Y = np.array(Yn)
			self.save_model("abc")



	def get_layer_vote(self, layer, img):
		s = 0
		a = 0
		for feature in layer:
			x = feature.get_vote(img)
			# print(x)
			s += x*feature.weight
			a += feature.weight
		# print(s,a)
		if s > a/2:
			return True
		else:
			return False

	def save_model(self, path):
		f = open(path, "wb")
		pickle.dump(self, f)
	def predict(self,X):
		if all(self.get_layer_vote(layer, X) == 1 for layer in self.model):
			return True
		return False
