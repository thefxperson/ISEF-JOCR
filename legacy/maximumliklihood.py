import idx2numpy as idx
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

labels = idx.convert_from_file("MNIST_data/train-labels.idx1-ubyte")
images = idx.convert_from_file("MNIST_data/train-images.idx3-ubyte")
testLabels = idx.convert_from_file("MNIST_data/t10k-labels.idx1-ubyte")
testImages = idx.convert_from_file("MNIST_data/t10k-images.idx3-ubyte")

#images [images number][x][y]
#labels [label number]

def norm_dist(x,std,mu):	#find probability that a point(x) is in the gaussian dist of that pixel (given its std and mu)
	'''expo = (-1*np.power((x-mu),2))/(2*np.power(std,2))
	#print(expo)
	coef = 1/(np.sqrt(2*np.pi)*std)
	#print(coef)
	#print(coef*np.exp(expo))
	return coef*np.exp(expo)'''
	val = x-mu
	val = np.power(val,2)
	denom = np.power(std,2)
	denom *= 2
	if denom == 0.0:
		return 0.0
	val *= -1
	val = val/denom
	val = np.exp(val)
	denom = 2*np.pi
	denom = np.sqrt(denom)
	denom *= std
	if denom == 0.0:
		return 0.0
	return val/denom

def clear_foo(foo):		#fills the foo array with NAN, making it clear which values were not filled in for loop, and makes masking them easy
	for i in range(0,10):
		for j in range(0,images.shape[0]):
			foo[i][j] = np.nan

	return foo

def writeVals(std,mu):		#saves computed std and mu for each pixel so computation doesn't need to be done each time
	np.save("MNIST_data/std.npy",std)
	np.save("MNIST_data/mu.npy",mu)

def compute():	#calculates the std and mu for each pixel from the training images
	std_data = np.empty([10,28,28])#[catagory][x][y]
	mu_data = np.empty([10,28,28])
	foo = np.empty([10,images.shape[0]])

	foo = clear_foo(foo)

	for i in range(0, images.shape[1]):
		for j in range(0, images.shape[2]):
			for k in range(0, images.shape[0]):
				 foo[labels[k]][k] = images[k][i][j]	#fills a buffer(foo) with the value of each pixel one at a time.

			#mx = ma.masked_values(foo, np.nan)	#mx is an array without -inf(numbers that weren't filled in)
			mx = ma.masked_invalid(foo)					#array created too large on purpose, this removes any values not filled (there aren't the same ammount of each number, probably)

			for l in range(0, 10):	#allow calculation for each number ... goes through the buffer for each number and calculates std and mu, stores them in respectiev arrays
				mu_data[l][i][j] = np.mean(mx[l])	#mean of each pixel
				std_data[l][i][j] = np.std(mx[l])	#std of each pixel

			foo = clear_foo(foo)	#clear the buffer array...probably not necesary on second thought

	writeVals(std_data,mu_data)			#stores computated values for future use

std_data = np.load("MNIST_data/std.npy")	#loads computed values
mu_data = np.load("MNIST_data/mu.npy")


#show the images (first arg)

'''for i in range(1,11):
	plt.subplot(2,10,i)
	plt.imshow(std_data[i-1],cmap=plt.cm.gray_r,interpolation="nearest")
	plt.axis("off")
	plt.subplot(2,10,i+10)
	plt.imshow(mu_data[i-1],cmap=plt.cm.gray_r,interpolation="nearest")
	plt.axis("off")

plt.show()'''

def calcProb(imgNum,numType):
	foo = 0
	for i in range(0,28):		#finds probability for each pixel, mults all prob together to find final prob--mult didn't work, finds mean probability
		for j in range(0,28):
			if norm_dist(testImages[imgNum][i][j],std_data[numType][i][j],mu_data[numType][i][j]) == 0.0:
				continue;
			else:
				foo+=norm_dist(testImages[imgNum][i][j],std_data[numType][i][j],mu_data[numType][i][j])
				#foo *= norm_dist(testImages[imgNum][i][j],std_data[numType][i][j],mu_data[numType][i][j])

	return foo/784

def choose(imgNum):		#chooses # with higest prob
	fooN = 0; fooL = 0
	for i in range (0,9):
		if calcProb(imgNum,i) > fooN:
			fooN = calcProb(imgNum,i)
			fooL = i
	return fooL

accuracy = 0
for i in range(0, len(testLabels)):
	if choose(i) == testLabels[i]:
		accuracy += 1
	print(i)

accuracy /= len(testLabels)

print(accuracy)