import numpy as np
import scipy.ndimage as sp
import scipy.misc as spmc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from datetime import datetime

class imageManager():
	def __init__(self):
		random.seed(datetime.now())						#seed based on current time... ensures that each episode is random

	def importImgs(self, classID, imgNum, imgSize=105):			#import 
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

		if imgNum < 10:
			loc = classID + "_0" + str(imgNum) + ".png"
		else:
			loc = classID + "_" + str(imgNum) + ".png"
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
			

	def getEpisode(self, numClasses, batch_size=16, numOutputs=30, numSamples=10, imgSize=105, train=True):
		random.seed(datetime.now())				#seed based on current time... ensures that each episode is random
		#prep for test or train episode
		if train:				#training class range...see Omniglot_data/README.md
			lower = 1
			upper = 964
		else:					#testing class range
			lower = 965
			upper = 1623

		classList = random.sample(range(lower,upper+1),numClasses)	#generate random classes totalling to numClasses

		#arrays of images and labels for the episode
		episodeImgs = np.zeros((batch_size, numSamples*numClasses, 400))		#array of zeros to hold images
		episodeLabels = np.empty((batch_size, numSamples*numClasses, numOutputs),dtype=object)		#empty array to hold labels
		alpha = [self.alphaHot(numClasses) for i in range(batch_size)]			#generate the random alphahot labels to use for this ep
		rotation = [[random.randint(0,3) for i in range(numClasses)] for j in range(batch_size)]	#rotates each class randomly by 90deg.
		imgNums = [random.randint(1,20) for i in range(numSamples*numClasses)]#temp random.sample(range(20),numSamples)						#generates numSamples unique numbers for the image so the same image isn't used twice
		inst = np.zeros((batch_size, numClasses, 3), dtype=int)							#stores the 1st, 5th, and 10th instance of every class
		foo = np.zeros((batch_size, numClasses))

		chooseClass = [[random.randint(0,numClasses-1) for i in range(numSamples*numClasses)] for j in range(batch_size)]#choose classes randomly
		for j in range(batch_size):
			for i in range(numSamples*numClasses):
				episodeImgs[j][i] = self.adjustImg(sp.interpolation.rotate(self.importImgs(classList[chooseClass[j][i]],imgNums[i]), (rotation[j][chooseClass[j][i]]*90))).flatten()	#get random image from chosen class, rotates that by 0, 90, 180, or 270 deg. 
																																					#Then randomly shifts and rotates by +-10px/+-10deg, and downscale to 20x20
				episodeLabels[j][i] = self.alphaToFive(alpha[j][chooseClass[j][i]])				#encodes label with alpha hot

				foo[j][chooseClass[j][i]] += 1
				if foo[j][chooseClass[j][i]] == 1:
					inst[j][chooseClass[j][i]][0] = i
				elif foo[j][chooseClass[j][i]] == 5:
					inst[j][chooseClass[j][i]][1] = i
				elif foo[j][chooseClass[j][i]] == 10:
					inst[j][chooseClass[j][i]][2] = i


		episodeLabels = np.reshape(episodeLabels, (batch_size*numClasses*numSamples, numOutputs))
		inst = np.reshape(inst, (batch_size*numClasses, 3))

		return episodeImgs, episodeLabels, inst				#returns images and labelss

	#generate unique labels for given number of classes in alphahot encoding
	def alphaHot(self, numClasses):
		order = random.sample(range(15625),numClasses)	#chooses unique random labels of the number of classes		15625 is the largest value possible with 6 letters
		hot = []
		for i in range(numClasses):
			baseFive = self.changeBase(order[i])		#changeBase returns an array, thus is called 1x per loop to get new array
			foo = ""
			for j in range(6):
				foo += chr(97+baseFive[j])					#change the base from 10 to 5 so it can be encoded using letters a->e
			hot.append(foo)
		return hot

	def changeBase(self, number):	#changes a number in base 10 to a number in base 5
		remainder = []
		for i in range(6):
			remainder.append(number % 5)	#mod func returns remainder
			number = int(number / 5)
		return remainder[::-1]			#reverse nums

	def adjustImg(self, image):
		image = sp.interpolation.rotate(image, (random.random()-.5)*20)		#rotate image between -10 and 10 degrees. (random returns between 0,1)
		image = sp.interpolation.shift(image, random.randint(-10,10))		#shift image between -10 and 10 pixels
		return spmc.imresize(image, (20,20))

	def alphaToFive(self, labels):            #predictions[30], labels[6]
		#convert from alpha to one hot
		one = [[None]*5 for _ in range(6)] 
		for i in range(len(labels)):         #converts to labels[6][5]
			if labels[i] == "a":
				one[i] = [1,0,0,0,0]
			elif labels[i] == "b":
				one[i] = [0,1,0,0,0]
			elif labels[i] == "c":
				one[i] = [0,0,1,0,0]
			elif labels[i] == "d":
				one[i] = [0,0,0,1,0]
			elif labels[i] == "e":
				one[i] = [0,0,0,0,1]
		return np.reshape(one, 30)