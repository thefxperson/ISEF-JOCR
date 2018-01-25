import tensorflow as tf
import numpy as np
import utils

#based on code from hmishra2250, used under MIT License. github: https://github.com/hmishra2250/NTM-One-Shot-TF

def MANN(input_var, target, batch_size=10, num_outputs=30, memory_shape=(128,40), controller_size=200, input_size=20*20, num_reads=4, num_samples_per_class=10, num_classes=5, firstTime=False):
	#input dims (batch_size, time, input_dim)
	#target dims (batch_size, time)(label_indicies)
	input_var = tf.reshape(input_var, [batch_size, num_classes*num_samples_per_class, 400])
	target = tf.reshape(target, [batch_size, num_classes*num_samples_per_class, num_outputs])

	memory = utils.shared_float32(1e-6*np.ones((batch_size,) + memory_shape), name="memory")
	cell_state = utils.shared_float32(np.zeros((batch_size, controller_size)), name="cell_state")
	hidden_state = utils.shared_float32(np.zeros((batch_size, controller_size)), name="hidden_state")
	read_vector = utils.shared_float32(np.zeros((batch_size, num_reads*memory_shape[1])), name="read_vector")
	read_weight_vector = utils.shared_one_hot((batch_size, num_reads, memory_shape[0]), name="read_weight_vector")
	usage_weights = utils.shared_one_hot((batch_size, memory_shape[0]), name="usage_weights")

	def shape_high(shape):          #sets up stuff for reading/writing to memory...the shape of the weights based off mem stuff and the range that the weights will be randomly initialized in
		shape = np.array(shape)
		if isinstance(shape, int):
			high = np.sqrt(6. / shape)
		else:
			high = np.sqrt(6. / (np.sum(shape[:2]) * np.prod(shape[2:])))
		return (list(shape), high)

	if firstTime:
		with tf.variable_scope("weights"):
			#get the weights and biases if they exist -- otherwise initialize weights and biases
			shape, high = shape_high((num_reads, controller_size, memory_shape[1]))
			weight_key = tf.get_variable("weight_key", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_key = tf.get_variable("bias_key", shape=(num_reads, memory_shape[1]), initializer=tf.constant_initializer(0))
			weight_alpha = tf.get_variable("weight_alpha", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_alpha = tf.get_variable("bias_alpha", shape=(num_reads, memory_shape[1]), initializer=tf.constant_initializer(0))

			shape, high = shape_high((num_reads, controller_size, 1))
			weight_sigma = tf.get_variable("weight_sigma", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_sigma = tf.get_variable("bias_sigma", shape=(num_reads, 1), initializer=tf.constant_initializer(0))

			shape, high = shape_high((input_size+num_outputs, 4*controller_size))
			weight_inputhidden = tf.get_variable("weight_inputhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_inputhidden = tf.get_variable("bias_inputhidden", shape=(4*controller_size), initializer=tf.constant_initializer(0))

			shape, high = shape_high((controller_size + num_reads * memory_shape[1], num_outputs))
			weight_output = tf.get_variable("weight_output", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_output = tf.get_variable("bias_output", shape=(num_outputs), initializer=tf.constant_initializer(0))

			shape, high = shape_high((num_reads * memory_shape[1], 4 * controller_size))
			weight_readhidden = tf.get_variable("weight_readhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))

			shape, high = shape_high((controller_size, 4*controller_size))
			weight_hiddenhidden = tf.get_variable("weight_hiddenhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))

			gamma = tf.get_variable("gamma", shape=[1], initializer=tf.constant_initializer(0.95))
	else:
		with tf.variable_scope("weights", reuse=True):
			#get the weights and biases if they exist -- otherwise initialize weights and biases
			shape, high = shape_high((num_reads, controller_size, memory_shape[1]))
			weight_key = tf.get_variable("weight_key", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_key = tf.get_variable("bias_key", shape=(num_reads, memory_shape[1]), initializer=tf.constant_initializer(0))
			weight_alpha = tf.get_variable("weight_alpha", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_alpha = tf.get_variable("bias_alpha", shape=(num_reads, memory_shape[1]), initializer=tf.constant_initializer(0))

			shape, high = shape_high((num_reads, controller_size, 1))
			weight_sigma = tf.get_variable("weight_sigma", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_sigma = tf.get_variable("bias_sigma", shape=(num_reads, 1), initializer=tf.constant_initializer(0))

			shape, high = shape_high((input_size+num_outputs, 4*controller_size))
			weight_inputhidden = tf.get_variable("weight_inputhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_inputhidden = tf.get_variable("bias_inputhidden", shape=(4*controller_size), initializer=tf.constant_initializer(0))

			shape, high = shape_high((controller_size + num_reads * memory_shape[1], num_outputs))
			weight_output = tf.get_variable("weight_output", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
			bias_output = tf.get_variable("bias_output", shape=(num_outputs), initializer=tf.constant_initializer(0))

			shape, high = shape_high((num_reads * memory_shape[1], 4 * controller_size))
			weight_readhidden = tf.get_variable("weight_readhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))

			shape, high = shape_high((controller_size, 4*controller_size))
			weight_hiddenhidden = tf.get_variable("weight_hiddenhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))

			gamma = tf.get_variable("gamma", shape=[1], initializer=tf.constant_initializer(0.95))

	def slice_equally(x, size, num_slices):
		#type: (object, object, object) -> object
		return [x[:,n*size:(n+1)*size] for n in range(num_slices)]

	def step(input_time1, input_time):  #network
		#values of time-1 for the network to use
		memory_time1, cell_time1, hidden_time1, read_time1, read_vector_time1, usage_weights_time1 = input_time1

		#get weights and biases
		with tf.variable_scope("weights", reuse=True):
			weight_key = tf.get_variable("weight_key", shape=(num_reads, controller_size, memory_shape[1]))
			bias_key = tf.get_variable("bias_key", shape=(num_reads, memory_shape[1]))
			weight_alpha = tf.get_variable("weight_alpha", shape=(num_reads, controller_size, memory_shape[1]))
			bias_alpha = tf.get_variable("bias_alpha", shape=(num_reads, memory_shape[1]))
			weight_sigma = tf.get_variable("weight_sigma", shape=(num_reads, controller_size, 1))
			bias_sigma = tf.get_variable("bias_sigma", shape=(num_reads, 1))

			weight_inputhidden = tf.get_variable("weight_inputhidden", shape=(input_size + num_outputs, 4 * controller_size))
			bias_inputhidden = tf.get_variable("bias_inputhidden", shape=(4 * controller_size))
			weight_output = tf.get_variable("weight_output", shape=(controller_size + num_reads * memory_shape[1], num_outputs))
			bias_output = tf.get_variable("bias_output", shape=(num_outputs))
			weight_readhidden = tf.get_variable("weight_readhidden", shape=(num_reads * memory_shape[1], 4 * controller_size))
			weight_hiddenhidden = tf.get_variable("weight_hiddenhidden", shape=(controller_size, 4 * controller_size))
			gamma = tf.get_variable("gamma", shape=[1], initializer=tf.constant_initializer(0.95))

		#lstm
		preactivations = tf.matmul(input_time, weight_inputhidden) + tf.matmul(read_time1, weight_readhidden) + tf.matmul(hidden_time1, weight_hiddenhidden) + bias_inputhidden		#input gate, forget gate, and output gate before they go through activation function
		forget_gate, input_gate, output_gate, inputtMod_gate = slice_equally(preactivations, controller_size, 4)
		#run values through activation functions
		forget_gate = tf.sigmoid(forget_gate)
		input_gate = tf.sigmoid(input_gate)
		output_gate = tf.sigmoid(output_gate)
		inputtMod_gate = tf.sigmoid(inputtMod_gate)
		#update states
		cell_time = forget_gate * cell_time1 + input_gate * inputtMod_gate	#update cell state
		hidden_time = output_gate * tf.tanh(cell_time)						#update hidden state


		#MANN

		head_param_list = tf.nn.xw_plus_b(hidden_time, weight_key, bias_key)
		head_param_list = tf.split(head_param_list, num_reads, axis=0)

		for i, param in enumerate(head_param_list):
			with tf.variable_scope("addressing head %d", i):
				key = tf.tanh(param[:, 0:memory_shape[1]], name="key")	#eq13 i think
				sigmoid_alpha = tf.sigmoid(param[:, -1:], name="sigmoid_alpha")
				weight_right

		'''
		#MANN
		hidden_time_weight_key = tf.matmul(hidden_time, tf.reshape(weight_key, shape=(controller_size, -1)))			#hidden layer is multiplied by weights before being activated to create key
		key_time = tf.tanh(tf.reshape(hidden_time_weight_key, shape=(batch_size, num_reads, memory_shape[1])) + bias_key)		#previous value through activation function (with bias) to create key (k sub t in paper, used in eq17)
		#alpha param... "a dynamic scalar gatte parameter to interpolate between the weights." used in creating write weights. eq 22
		hidden_time_weight_alpha = tf.matmul(hidden_time, tf.reshape(weight_alpha, shape=(controller_size, -1)))
		alpha_time = tf.tanh(tf.reshape(hidden_time_weight_alpha, shape=(batch_size, num_reads, memory_shape[1])) + bias_alpha)
		#not from paper... sigma from eq22, was supposed to be a sigmoid function of alpha... i don't wanna mess with the shapes of everything at this point
		hidden_time_weight_sigma = tf.matmul(hidden_time, tf.reshape(weight_sigma, shape=(controller_size,-1)))
		sigma_time = tf.sigmoid(tf.reshape(hidden_time_weight_sigma, shape=(batch_size, num_reads, 1)) + bias_sigma)
		#used in calculating write weights
		foo, temp_indicies = tf.nn.top_k(usage_weights_time1, memory_shape[0])
		weight_leastused_time1 = tf.slice(temp_indicies, [0,0], [batch_size, num_reads])

		#sigma_time_read_vector_time1 = tf.tile(sigma_time, tf.stack([1,1, read_vector_time1.get_shape().as_list()[2]]))			useless?
		#find write weights
		weight_write_time = tf.reshape(sigma_time*read_vector_time1, (batch_size*num_reads, memory_shape[0]))
		weight_write_time = utils.update_tensor(weight_write_time, tf.reshape(weight_leastused_time1,[-1]), 1.0 - tf.reshape(sigma_time, shape=[-1]))
		weight_write_time = tf.reshape(weight_write_time, (batch_size, num_reads, memory_shape[0]))

		#prep memory
		with tf.variable_scope("memory_time"):
			memory_time = utils.update_tensor(memory_time1, weight_leastused_time1[:,0], tf.constant(0., shape=[batch_size, memory_shape[1]]))
			#print("mem time (erased?): ", tf.eval(memory_time))


		memory_time = tf.add(memory_time, tf.matmul(tf.transpose(weight_write_time, perm=[0,2,1]), alpha_time))
		key_time = utils.cosine_similarity(key_time, memory_time)

		read_vector_time = tf.nn.softmax(tf.reshape(key_time, (batch_size*num_reads, memory_shape[0])))
		read_vector_time = tf.reshape(read_vector_time, (batch_size, num_reads, memory_shape[0]))

		usage_weights_time = gamma * usage_weights_time1 + tf.reduce_sum(read_vector_time, axis=1) + tf.reduce_sum(weight_write_time, axis=1)

		read_time = tf.reshape(tf.matmul(read_vector_time, memory_time), [batch_size, -1])

		return [memory_time, cell_time, hidden_time, read_time, read_vector_time, usage_weights_time]
		'''

	#model
	sequence_length = target.get_shape().as_list()[1]
	output_shape = (batch_size*sequence_length, num_outputs)

	#input concat with time offset
	#flattened_onehot_target = tf.one_hot(tf.reshape(target, [-1]), depth=num_outputs)
	#onehot_target = tf.reshape(flattened_onehot_target, (batch_size, sequence_length, num_outputs))
	#flattened_target = tf.reshape(target, [-1])

	offset_target = tf.concat([tf.zeros_like(tf.expand_dims(target[:,0,:],1)), target[:,:-1,:]], axis=1)
	list_input = tf.concat([input_var, offset_target], axis=2)

	list_ntm = tf.scan(step, elems=tf.transpose(list_input, perm=[1,0,2]), initializer=[memory, cell_state, hidden_state, read_vector, read_weight_vector, usage_weights], name="Scan_MANN_last")
	list_ntm_output = tf.transpose(tf.concat(list_ntm[2:4], axis=2), perm=[1,0,2])

	list_input_weight_output = tf.matmul(tf.reshape(list_ntm_output, shape=(batch_size* sequence_length, -1)), weight_output)
	output_preactivation = tf.add(tf.reshape(list_input_weight_output, shape=(batch_size, sequence_length, num_outputs)), bias_output)
	output_flatten = tf.reshape(output_preactivation, output_shape)
	output_flatten = tf.split(output_flatten, 6, axis=1)
	output_flatten = tf.nn.softmax(output_flatten)
	output = tf.reshape(output_flatten, output_shape)
	output_flatten = tf.argmax(output_flatten, axis=2)
	output_flatten = tf.one_hot(output_flatten, 5)
	output_flatten = tf.reshape(output_flatten, output_shape)
	#output distribution (but its only one hot)
	#output = tf.stack([tf.nn.softmax(o) for o in [tf.split(p , 5, axis=1) for p in tf.split(output_flatten, 6, axis=1)]], axis=1)
	#output = tf.reshape(output, output_shape)

	params = [weight_key, bias_key, weight_alpha, bias_alpha, weight_sigma, bias_sigma, weight_inputhidden, weight_readhidden, weight_hiddenhidden, bias_inputhidden, weight_output, bias_output]

	return output, output_flatten, params