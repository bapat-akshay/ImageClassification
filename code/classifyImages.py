import tensorflow as tf
import cv2
from natsort import natsorted
import csv
import os


testPath = "test1/"
name = "Classification-1589476013"

def prepareTestImg(filepath):
	size = 75
	img = cv2.imread(filepath)
	img = cv2.resize(img, (size, size))

	return img.reshape(-1, size, size, 3)


model = tf.keras.models.load_model("{}.h5".format(name))
total = 0
correct = 0
answerKey = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1,
 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 
 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 
 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 
 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 
 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 
 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 
 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 
 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 
 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 
 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 
 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]

with open("output.csv", mode='a+', newline='') as out:
	imgList = []
	csv_writer = csv.writer(out)
	
	for file in natsorted(os.listdir(testPath), key=lambda y:y.lower()):
		#imgList.append(prepareTestImg(os.path.join(testPath, file)))
	
		pred = model.predict([prepareTestImg(os.path.join(testPath, file))])
		#print(pred)
		csv_writer.writerow([file, int(pred[0][0])])
		if int(pred[0][0]) == answerKey[total]:
			correct += 1
		total += 1
		if total == len(answerKey):
			break	

corrPerc = correct*100/total
print("Accuracy for {} test images: {}%".format(len(answerKey), corrPerc))