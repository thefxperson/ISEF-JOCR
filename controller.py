import numpy as np
import idx2numpy as idx

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils

labels = idx.convert_from_file("MNIST_data/train-labels.idx1-ubyte")
images = idx.convert_from_file("MNIST_data/train-images.idx3-ubyte")
testLabels = idx.convert_from_file("MNIST_data/t10k-labels.idx1-ubyte")
testImages = idx.convert_from_file("MNIST_data/t10k-images.idx3-ubyte")

images = np.reshape(images,(60000,784))
images = np.expand_dims(images, axis=2)
labels = np_utils.to_categorical(labels, 10)

testImages = np.reshape(testImages,(10000,784))
testImages = np.expand_dims(testImages, axis=2)
testLabels = np_utils.to_categorical(testLabels,10)

def  createModel():#create the model
	model = Sequential()
	model.add(LSTM(200, input_shape=images.shape[1:],forget_bias_init="one",activation="tanh",inner_activation="sigmoid"))
	model.add(Dense(10))
	model.add(Activation("softmax"))
	model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
	model.fit(images, labels, nb_epoch=6, batch_size=100, verbose=1, validation_data=(testImages,testLabels))

	model.save("LSTMModel.h5")

#createModel()

model = load_model("LSTMModel.h5")
scores = model.evaluate(testImages, testLabels, verbose=0)
print("Test Loss: ", scores[0])
print("Test Accuracy: ", scores[1])