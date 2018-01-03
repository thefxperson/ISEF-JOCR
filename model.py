import tensorflow as tf
import numpy as np
#import imageManager
import utils

'''mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

nm_episodes = 100000		#number of episodes
nm_classes = 30		#number of outputs
batch_size = 100	#number of images per batch
chunk_size = 28		#image x size
nm_chunks = 28		#image y size
hdn_units = 200		#number of hidden units

#placeholder variables for images and labels
x = tf.placeholder("float", [None, nm_chunks, chunk_size])
y = tf.placeholder("float")

manager = imageManager.imageManager()

#lstm layer of network
def lstm(x, name="lstm"):
	with tf.name_scope(name):
		#dictionary that initializes weights and biases randomly
		hidden_layer = {"weights":tf.Variable(tf.random_normal([hdn_units,nm_classes])),
						"biases":tf.Variable(tf.random_normal([nm_classes]))}

		#resize data shape for network
		x = tf.transpose(x, [1,0,2])
		x = tf.reshape(x,[-1, chunk_size])
		x = tf.split(x, nm_chunks, 0)

		#create lstm cells
		lstm_cell = rnn_cell.BasicLSTMCell(hdn_units,state_is_tuple=True)
		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)	#get output of network

		#multiply output by weights + biases
		output = tf.matmul(outputs[-1],hidden_layer["weights"]) + hidden_layer["biases"]
		
		#add data to tensorboard
		tf.summary.histogram("weights", hidden_layer["weights"])
		tf.summary.histogram("biases", hidden_layer["biases"])
		tf.summary.histogram("lstm_output", output)

		return output


#train the neural network
def train_NN(x):
	with tf.name_scope("train"):
		prediction = lstm(x)	#get output from lstm
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))	#calculate loss
		optimizer = tf.train.RMSPropOptimizer(1e-4, decay=0.95, momentum=0.9).minimize(cost)		#minimize loss with RMSProp, learning rate, momentum, and decay from DeepMind's MANN paper

		tf.summary.scalar("cost", cost)

	saver = tf.train.Saver()	#create a savepoint of the model

	#begin the training
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())	#init the variables
		merged = tf.summary.merge_all()
		global summary_val

		#run through all epochs
		for episode in range(nm_episodes):
			episode_loss = 0	#loss in current episode
			episode_x, episode_y =  manager.getEpisode(5+int((episode*10)/nm_episodes))		#pull episode info: images and alphahot labels
			episode_x = episode_x.reshape((batch_size,nm_chunks,chunk_size))		#reshape results

			i, c, summary_val = sess.run([optimizer, cost, merged], feed_dict={x:episode_x,y:episode_y})	#backpropigate using optimizer, update weights,ect

			episode_loss += c 		#cost for epoch

			print("Episode ", episode+1, " completed out of ", nm_episodes, " loss: ", episode_loss)

		#test the accuracy
		with tf.name_scope("accuracy"):
			correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
			accuracy = tf.reduce_mean(tf.cast(correct, "float"))
			tf.summary.scalar("accuracy", accuracy)
		print("accuracy: ", accuracy.eval({x:mnist.test.images.reshape((-1,nm_chunks,chunk_size)),y:mnist.test.labels}))
		saver.save(sess, "/save/lstmSave.ckpt")		#save learned weights and biases
		#for tensorboard
		writer = tf.summary.FileWriter("/tmp/tboard/")
		writer.add_summary(summary_val)
		writer.add_graph(sess.graph)

def getLoss(prediction, labels):            #predictions[30], labels[6]
	#convert from alpha to one hot
	for j in range(len(labels)):         #converts to labels[6][5]
		if labels[i][j] == "a":
			one[i][j] = [1,0,0,0,0]
		elif labels[i][j] == "b":
			one[i][j] = [0,1,0,0,0]
		elif labels[i][j] == "c":
			one[i][j] = [0,0,1,0,0]
		elif labels[i][j] == "d":
			one[i][j] = [0,0,0,1,0]
		elif labels[i][j] == "e":
			one[i][j] = [0,0,0,0,1]
	one = one.reshape(one, 30)
	#get loss
	diff = one - prediction
	loss = -1*np.log10(diff)        #log0 = Nan
	loss[np.isnan(loss)] = 0        #replace Nan with 0
	return np.sum(loss)

#train_NN(x)
print(int(mnist.train.num_examples/batch_size))

lstm(x)	#create variables before loading
with tf.Session() as sess:
	saver = tf.train.Saver()	#load learned weights+biases
	saver.restore(sess, "./lstmSave.ckpt")'''

#based on code from hmishra2250, used under MIT License. github: https://github.com/hmishra2250/NTM-One-Shot-TF

def MANN(input_var, target, batch_size=16, num_classes=5, memory_shape=(128,40), controller_size=200, input_size=20*20, num_reads=4):
	#input dims (batch_size, time, input_dim)
	#target dims (batch_size, time)(label_indicies)

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

	with tf.variable_scope("weights"):
		#get the weights and biases if they exist -- otherwise initialize weights and biases
		shape, high = shape_high((num_reads, controller_size, memory_shape[1]))
		weight_key = tf.get_variable("weight_key", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
		bias_key = tf.get_variable("bias_key", shape=(num_reads, memory_shape[1]), initializer=tf.constant_initializer(0))
		weight_add = tf.get_variable("weight_add", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
		bias_add = tf.get_variable("bias_add", shape=(num_reads, memory_shape[1]), initializer=tf.constant_initializer(0))

		shape, high = shape_high((num_reads, controller_size, 1))
		weight_sigma = tf.get_variable("weight_sigma", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
		bias_sigma = tf.get_variable("bias_sigma", shape=(num_reads, 1), initializer=tf.constant_initializer(0))

		shape, high = shape_high((input_size+num_classes, 4*controller_size))
		weight_inputhidden = tf.get_variable("weight_inputhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
		bias_inputhidden = tf.get_variable("bias_inputhidden", shape=(4*controller_size), initializer=tf.constant_initializer(0))

		shape, high = shape_high((controller_size + num_reads * memory_shape[1], num_classes))
		weight_output = tf.get_variable("weight_output", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))
		bias_output = tf.get_variable("bias_output", shape=(num_classes), initializer=tf.constant_initializer(0))

		shape, high = shape_high((num_reads * memory_shape[1], 4 * controller_size))
		weight_readhidden = tf.get_variable("weight_readhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))

		shape, high = shape_high((controller_size, 4*controller_size))
		weight_hiddenhidden = tf.get_variable("weight_hiddenhidden", shape=shape, initializer=tf.random_uniform_initializer(-1*high, high))

		gamma = tf.get_variable("gamma", shape=[1], initializer=tf.constant_initializer(0.95))

	def slice_equally(x, size, num_slices):
		#type: (object, object, object) -> object
		return [x[:,n*size:(n+1)*size] for n in range(num_slices)]

	def step(input_time1, input_time):  #network
		memory_time1, cell_time1, hidden_time1, read_time1, read_vector_time1, usage_weights_time1 = input_time1

		with tf.variable_scope("weights", reuse=True):
			weight_key = tf.get_variable("weight_key", shape=(num_reads, controller_size, memory_shape[1]))
			bias_key = tf.get_variable("bias_key", shape=(num_reads, memory_shape[1]))
			weight_add = tf.get_variable("weight_add", shape=(num_reads, controller_size, memory_shape[1]))
			bias_add = tf.get_variable("bias_add", shape=(num_reads, memory_shape[1]))
			weight_sigma = tf.get_variable("weight_sigma", shape=(num_reads, controller_size, 1))
			bias_sigma = tf.get_variable("bias_sigma", shape=(num_reads, 1))
			weight_inputhidden = tf.get_variable("weight_inputhidden", shape=(input_size + num_classes, 4 * controller_size))
			bias_inputhidden = tf.get_variable("bias_inputhidden", shape=(4 * controller_size))
			weight_output = tf.get_variable("weight_output", shape=(controller_size + num_reads * memory_shape[1], num_classes))
			bias_output = tf.get_variable("bias_output", shape=(num_classes))
			weight_readhidden = tf.get_variable("weight_readhidden", shape=(num_reads * memory_shape[1], 4 * controller_size))
			weight_hiddenhidden = tf.get_variable("weight_hiddenhidden", shape=(controller_size, 4 * controller_size))
			gamma = tf.get_variable("gamma", shape=[1], initializer=tf.constant_initializer(0.95))

		preactivations = tf.matmul(input_time, weight_inputhidden) + tf.matmul(read_time1, weight_readhidden) + tf.matmul(hidden_time1, weight_hiddenhidden) + bias_inputhidden
		forget_gate, input_gate, output_gate, u = slice_equally(preactivations, controller_size, 4)
		forget_gate = tf.sigmoid(forget_gate)
		input_gate = tf.sigmoid(input_gate)
		output_gate = tf.sigmoid(output_gate)
		u = tf.sigmoid(u)

		cell_time = forget_gate * cell_time1 + input_gate * u
		hidden_time = output_gate * tf.tanh(cell_time)

		hidden_time_weight_key = tf.matmul(hidden_time, tf.reshape(weight_key, shape=(controller_size, -1)))
		key_time = tf.tanh(tf.reshape(hidden_time_weight_key, shape=(batch_size, num_reads, memory_shape[1])) + bias_key)
		hidden_time_weight_add = tf.matmul(hidden_time, tf.reshape(weight_add, shape=(controller_size, -1)))
		add_time = tf.tanh(tf.reshape(hidden_time_weight_add, shape=(batch_size, num_reads, memory_shape[1])) + bias_add)
		hidden_time_weight_sigma = tf.matmul(hidden_time, tf.reshape(weight_sigma, shape=(controller_size,-1)))
		sigma_time = tf.sigmoid(tf.reshape(hidden_time_weight_sigma, shape=(batch_size, num_reads, 1)) + bias_sigma)

		foo, temp_indicies = tf.nn.top_k(usage_weights_time1, memory_shape[0])

		weight_leastused_time1 = tf.slice(temp_indicies, [0,0], [batch_size, num_reads])

		sigma_time_read_vector_time1 = tf.tile(sigma_time, tf.stack([1,1, read_vector_time1.get_shape().as_list()[2]]))
		weight_write_time = tf.reshape(sigma_time*read_vector_time1, (batch_size*num_reads, memory_shape[0]))
		weight_write_time = utils.update_tensor(weight_write_time, tf.reshape(weight_leastused_time1,[-1]), 1.0 - tf.reshape(sigma_time, shape=[-1]))
		weight_write_time = tf.reshape(weight_write_time, (batch_size, num_reads, memory_shape[0]))

		with tf.variable_scope("memory_time"):
			print("Weights (least-used) (time+1): ", weight_leastused_time1.get_shape().as_list())
			memory_time = utils.update_tensor(memory_time1, weight_leastused_time1[:,0], tf.constant(0., shape=[batch_size, memory_shape[1]]))
		memory_time = tf.add(memory_time, tf.matmul(tf.transpose(weight_write_time, perm=[0,2,1]), add_time))
		key_time = utils.cosine_similarity(key_time, memory_time)

		read_vector_time = tf.nn.softmax(tf.reshape(key_time, (batch_size*num_reads, memory_shape[0])))
		read_vector_time = tf.reshape(read_vector_time, (batch_size, num_reads, memory_shape[0]))

		usage_weights_time = gamma * usage_weights_time1 + tf.reduce_sum(read_vector_time, axis=1) + tf.reduce_sum(weight_write_time, axis=1)

		read_time = tf.reshape(tf.matmul(read_vector_time, memory_time), [batch_size, -1])

		return [memory_time, cell_time, hidden_time, read_time, read_vector_time, usage_weights_time]

	#model
	sequence_length = target.get_shape().as_list()[1]
	output_shape = (batch_size*sequence_length, num_classes)

	#input concat with time offset
	flattened_onehot_target = tf.one_hot(tf.reshape(target, [-1]), depth=num_classes)
	onehot_target = tf.reshape(flattened_onehot_target, (batch_size, sequence_length, num_classes))
	offset_target = tf.concat([tf.zeros_like(tf.expand_dims(onehot_target[:,0],1)), onehot_target[:,:-1]], axis=1)
	list_input = tf.concat([input_var, offset_target], axis=2)

	list_ntm = tf.scan(step, elems=tf.transpose(list_input, perm=[1,0,2]), initializer=[memory, cell_state, hidden_state, read_vector, read_weight_vector, usage_weights], name="Scan_MANN_last")
	list_ntm_output = tf.transpose(tf.concat(list_ntm[2:4], axis=2), perm=[1,0,2])

	list_input_weight_output = tf.matmul(tf.reshape(list_ntm_output, shape=(batch_size * sequence_length, -1)), weight_output)
	output_preactivation = tf.add(tf.reshape(list_input_weight_output, shape=(batch_size, sequence_length, num_classes)), bias_output)
	output_flatten = tf.nn.softmax(tf.reshape(output_preactivation, output_shape))
	output = tf.reshape(output_flatten, output_preactivation.get_shape().as_list())

	params = [weight_key, bias_key, weight_add, bias_add, weight_sigma, bias_sigma, weight_inputhidden, weight_readhidden, weight_hiddenhidden, bias_inputhidden, weight_output, bias_output]

	return output, output_flatten, params