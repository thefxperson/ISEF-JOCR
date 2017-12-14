import numpy as np
import scipy as sp
import matplotlib.image as mpimg
import random
from datetime import datetime

class imageManager():
	def __init__(self):
		random.seed(datetime.now())						#seed based on current time... ensures that each episode is random

	def importImgs(self,classID,imgSize=105):			#import 
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
		random.seed(datetime.now())				#seed based on current time... ensures that each episode is random
		#prep for test or train episode
		if train:				#training class range...see Omniglot_data/README.md
			lower = 0
			upper = 964
		else:					#testing class range
			lower = 965
			upper = 1623

		classList = random.sample(range(lower,upper+1),numClasses)	#generate random classes totalling to numClasses

		#arrays of images and labels for the episode
		episodeImgs = np.zeros((10*numClasses,20,20))		#array of zeros to hold images
		episodeLabels = np.empty(10*numClasses,dtype=object)		#empty array to hold labels
		alpha = self.alphaHot(numClasses)							#generate the random alphahot labels to use for this ep

		chooseClass = [random.randint(0,numClasses-1) for i in range(10*numClasses)]	#choose classes randomly
		print(chooseClass)

		for i in range(10*numClasses):
			episodeImgs[i] = self.importImgs(classList[chooseClass[i]])	#gets a random image from chosen class
			episodeLabels[i] = alpha[chooseClass]				#encodes label with alpha hot

		#return episodeImgs, episodeLabels				#returns images and labelss

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
		image = sp.ndimage.interpolation.rotate(image, (random.random()-.5)*20)		#rotate image between -10 and 10 degrees. (random returns between 0,1)
		image = sp.ndimage.interpolation.shift(image, random.randint(-10,10))		#shift image between -10 and 10 pixels
		return sp.misc.imresize(image, (20,20))


foo = imageManager()
foo.getEpisode(2)