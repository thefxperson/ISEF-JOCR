import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from datetime import datetime

class imageManager():
	def __init__(self):
		random.seed(datetime.now())

	def importImgs(self,classID,imgSize=105):
		#determine if testing image or training image
		if classID <= 964:
			imgPath = "images_background"
		else:
			imgPath = "images_evaluation"

		#prepare ID for use in path, convert to string
		if classID < 10:
			classID = "000" + str(classID)
		elif classID < 100:
			classID = "00" + str(classID)
		elif classID < 1000:
			classID = "0" + str(classID)
		else:
			classID = str(classID)

		#create array to hold image(s)
		imgArray = np.zeros((imgSize,imgSize))

		#load image
		i = random.randint(1,20)

		if i < 10:
			loc = classID + "_0" + str(i) + ".png"
		else:
			loc = loc = classID + "_" + str(i) + ".png"
		imgArray = self.cleanData(mpimg.imread("Omniglot_data/" + imgPath + "/" + loc))

		return imgArray
			

	def cleanData(self, images):	#flips 0s and 1s. this way 1 = black and 0 = white
		for i in range(images.shape[0]):
			for j in range(images.shape[1]):
					if images[i][j] == 1:
						images[i][j] = 0
					else:
						images[i][j] = 1
		return images
			

	def getEpisode(self, numClasses, imgSize=105, train=True):
		#prep for test or train episode
		if train:
			lower = 0
			upper = 964
		else:
			lower = 965
			upper = 1623

		classList = random.sample(range(lower,upper+1),numClasses)	#generate random classes totalling to numClasses

		#arrays of images and labels for the episode
		episodeImgs = np.zeros((10*numClasses,imgSize,imgSize))
		episodeLabels = np.zeros(10*numClasses)

		for i in range(10*numClasses):
			chooseClass = random.randint(0,numClasses-1)	#choose which class, returns var from 0-numClasses
			episodeImgs[i] = self.importImgs(classList[chooseClass])	#gets a random image from chosen class, using classDict
			'''episodeLabels[i] = alphaHot(chooseClass)	'''#encodes the label as alphaHot type
			episodeLabels[i] = classList[chooseClass]

		return episodeImgs, episodeLabels				#returns images and labelss


	#need x unique labels, where each label has 5 unqiue vals
	def alphaHot(self, classList):	#encodes label with alphahot
		order = random.sample(range(0,5),5)
		hot = ""
		for j in range(numClasses):				#convert to string (97 is lowercase a)
			hot += chr(97+order[j])
		print(hot)

imgM = imageManager()
#imgs, labels = imgM.getEpisode(5)