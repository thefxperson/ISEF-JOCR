import tensorflow as tf
import numpy as np
import pytest

import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage import rotate,shift
import matplotlib.image as mpimg
from scipy.misc import imresize

import imageManager

import os
import random

#the following code has been taken or adapted from hmishra2250's github under the MIT liscence. github link: https://github.com/hmishra2250/NTM-One-Shot-TF

class OmniglotGenerator(object):
	"""Docstring for OmniglotGenerator"""
	def __init__(self, data_folder, batch_size=1, num_samples=30, num_samples_per_class=10, max_rotation=-np.pi/6, max_shift=10, img_size=(20,20), max_iter=None, num_classes=5):
		super(OmniglotGenerator, self).__init__()
		self.data_folder = data_folder
		self.batch_size = batch_size
		self.num_samples = num_samples
		self.num_samples_per_class = num_samples_per_class
		self.max_rotation = max_rotation
		self.max_shift = max_shift
		self.img_size = img_size
		self.max_iter = max_iter
		self.num_iter = 0
		#self.character_folders = [os.path.join(self.data_folder, family, character) for family in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, family)) for character in os.listdir(os.path.join(self.data_folder, family))]
		self.img = imageManager.imageManager()
		self.num_classes = num_classes

	def __iter__(self):
		return self

	def __next__(self):
		return self.next()

	def next(self):
		if (self.max_iter is None) or (self.num_iter < self.max_iter):
			self.num_iter += 1
			return (self.num_iter - 1), self.img.getEpisode(self.num_classes, batch_size=self.batch_size)#self.sample(self.num_samples)
		else:
			raise StopIteration

	'''def sample(self, num_samples):
		sampled_character_folders = random.sample(self.character_folders, num_samples)
		random.shuffle(sampled_character_folders)

		images = self.img.getEpisode(self.num_classes, self.num_samples_per_class)

		example_inputs = np.zeros((self.batch_size, num_samples * self.num_samples_per_class, np.prod(self.img_size)), dtype=np.float32)
		example_outputs = np.zeros((self.batch_size, num_samples * self.num_samples_per_class), dtype=np.float32)     #notice hardcoded np.float32 here and above, change it to something else in tf

		for i in range(self.batch_size):
			labels_and_images = self.get_shuffled_images(sampled_character_folders, range(num_samples), num_samples=self.num_samples_per_class)
			sequence_length = len(labels_and_images)
			labels, image_files = zip(*labels_and_images)
			print(labels)
			print(image_files)
			angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=sequence_length)
			shifts = np.random.uniform(-self.max_shift, self.max_shift, size=sequence_length)

			example_inputs[i] = np.asarray([images[i].flatten() \
											for (filename, angle, shift) in zip(image_files, angles, shifts)], dtype=np.float32)
			example_outputs[i] = np.asarray(labels, dtype=np.int32)

		return example_inputs, example_outputs

	def get_shuffled_images(self, paths, labels, num_samples=None):
		if num_samples is not None:
			sampler = lambda x: random.sample(x, num_samples)
		else:
			sampler = lambda x:x

		images = [(i, os.path.join(path, image)) for i,path in zip(labels,paths) for image in sampler(os.listdir(path)) ]
		random.shuffle(images)
		return images

	def time_offset_label(self, labels_and_images):
		labels, images = zip(*labels_and_images)
		time_offset_labels = (None,) + labels[:-1]
		return zip(images, time_offset_labels)

	def load_transform(self, image_path, angle=0., s=(0,0), size=(20,20)):
		#Load the image
		original = mpimg.imread(image_path)
		#Rotate the image
		rotated = np.maximum(np.minimum(rotate(original, angle=angle, cval=1.), 1.), 0.)
		#Shift the image
		shifted = shift(rotated, shift=s)
		#Resize the image
		resized = np.asarray(imresize(rotated, size=size), dtype=np.float32) / 255 #Note here we coded manually as np.float32, it should be tf.float32
		#Invert the image
		inverted = 1. - resized
		max_value = np.max(inverted)
		if max_value > 0:
			inverted /= max_value
		return inverted'''