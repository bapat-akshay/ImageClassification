import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


path = "E:/Akshay/673_proj6/code/train"
categories = ["cat", "dog"]
IMG_SIZE = 75


def createTrainingData():
	train = []
	for c in categories:
		imgPath = os.path.join(path, c)
		for file in os.listdir(imgPath):
			try:
				img = cv2.imread(os.path.join(imgPath, file))
				img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
				train.append([img, categories.index(c)])
			except:
				print("Unable to read file " + file)
	random.shuffle(train)

	return train

print("Reading images...")
train = createTrainingData()
# for t in train[:5]:
# 	cv2.imshow("", t[0])
# 	print(t[1])
# 	cv2.waitKey(0)

# cv2.destroyAllWindows()

X = []
Y = []

for features, label in train:
	X.append(features)
	Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

print("Generating output pickle files...")
p = open("X.pickle", "wb")
pickle.dump(X, p)
p.close()

p = open("Y.pickle", "wb")
pickle.dump(Y, p)
p.close()

print("Done")