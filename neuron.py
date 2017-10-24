import math

class neuron:
	def __init__ (self, weights, bias):
		self.weights = weights
		self.bias = bias

	def fire(self, input, type):
		for i in range(len(input)):
			self.inputs.append(input[i]*self.weights[i])
		for j in range(len(self.inputs)):
			self.val += self.inputs[j]

		self.val += self.bias

		if type == "relu":
			return relu(self.val)
		elif type == "sig":
			return sigmoid(self.val)
		else:
			return 0

	def relu(val):
		if val <=0:
				return 0
			else:
				return val

	def sigmoid(val):
		return 1/(1+math.exp(-val))