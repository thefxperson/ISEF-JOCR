import csv
import imageManager
import numpy as np
'''with open("kana.csv", "r", encoding="utf-8") as f:
	r = csv.DictReader(f, delimiter=",", skipinitialspace=True)
	fooChar = [row["Char"] for row in r]

img = imageManager.imageManager()


def alphaToFive(labels):            #predictions[30], labels[6]
	#convert from alpha to one hot
	one = [[None]*5 for _ in range(6)] 
	for i in range(len(labels)):         #converts to labels[6][5]
		if labels[i] == 0:
			one[i] = [1,0,0,0,0]
		elif labels[i] == 1:
			one[i] = [0,1,0,0,0]
		elif labels[i] == 2:
			one[i] = [0,0,1,0,0]
		elif labels[i] == 3:
			one[i] = [0,0,0,1,0]
		elif labels[i] == 4:
			one[i] = [0,0,0,0,1]
	return np.reshape(one, 30)

#fooDict = {}
fooDict = np.load("kanji_list.npy").item()
for i in range(len(fooChar)):
	fooDict[fooChar[i]] = alphaToFive(img.changeBase(len(fooDict)+i))

np.save("kanji_list.npy", fooDict)'''

fooDict = np.load("kanji_list.npy").item()
print(fooDict["大"])
print(fooDict["好"])
print(fooDict["き"])