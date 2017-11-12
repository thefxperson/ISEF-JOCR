import idx2numpy as idx
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

labels = idx.convert_from_file("MNIST_data/train-labels.idx1-ubyte")
images = idx.convert_from_file("MNIST_data/train-images.idx3-ubyte")

#images [images number][x][y]
#labels [label number]

def norm_dist(x,std,mu):
	expo = (-1*np.power((x-mu),2))/(2*np.power(std,2))
	coef = 1/(np.sqrt(2*np.pi)*std)
	return coef*np.exp(expo)

def clear_foo(foo):
	for i in range(0,10):
		for j in range(0,100):
			foo[i][j] = np.nan

	return foo

def writeVals(std,mu):
	np.save("MNIST_data/std.npy",std)
	np.save("MNIST_data/mu.npy",mu)

dataset = np.array([1,2,3,4,5])


std_data = np.empty([10,28,28])#[catagory][x][y]
mu_data = np.empty([10,28,28])
foo = np.empty([10,100])

foo = clear_foo(foo)

for i in range(0, images.shape[1]):
	for j in range(0, images.shape[2]):
		for k in range(0, 100):
			 foo[labels[k]][k] = images[k][i][j]

		#mx = ma.masked_values(foo, np.nan)	#mx is an array without -inf(numbers that weren't filled in)
		mx = ma.masked_invalid(foo)

		for l in range(0, 10):	#allow calculation for each number
			mu_data[l][i][j] = np.mean(mx[l])	#mean of each pixel
			std_data[l][i][j] = np.std(mx[l])	#std of each pixel

		foo = clear_foo(foo)

writeVals(std_data,mu_data)

#show the image (first arg)
plt.imshow(std_data[9],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()