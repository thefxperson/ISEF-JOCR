import numpy as np
import random

def getEpisode(classes):

	classNums = np.zeros(15)
	for i in range(16):	#choose 15 classes
		classNums[i] = random.randint(0,classes.shape[0])

		for j in range(16):	#check for dupe classes
			if classNums[i] == classNums[j] and i != j:
				cclassNums[i] = random.randint(0,classes.shape[0])

	for i in range(61):
		ff

for i in range(15):
	print(i)