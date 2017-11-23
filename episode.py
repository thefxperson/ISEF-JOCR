import numpy as np
import random

def getEpisode(classes, numClasses, imgSize=105):

	classNums = np.zeros(numClasses)
	for i in range(numClasses):	#choose 15 classes to use
		classNums[i] = random.randint(0,classes.shape[0])

		for j in range(15):	#check for dupe classes
			if classNums[i] == classNums[j] and i != j:
				classNums[i] = random.randint(0,classes.shape[0])

		classDict = {i:classNums[i]}	#dict of incremental numbers to class num. ie 1 - 200, 2 - 659, 3 - 914, 4 - 142 ect

	#arrays of images and labels for the episode
	episodeImgs = np.zeros((10*numClasses,imgSize,imgSize))
	episodeLabels = np.zeros(10*numClasses)

	for i in range(10*numClasses):
		chooseClass = random.randint(0,numClasses)	#choose which class, returns var from 0-numClasses
		episodeImgs[i] = classes[classDict[chooseClass]][random.randint(0,20)]	#gets a random image from chosen class, using classDict
		episodeLabels[i] = alphaHot(chooseClass)	#encodes the label as alphaHot type

	return episodeImgs, episodeLabels				#returns images and labelss