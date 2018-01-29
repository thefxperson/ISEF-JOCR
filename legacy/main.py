import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #No logging TF

import tensorflow as tf
import numpy as np
import time

import utils
import model
import omniGenerator

#the following code has been taken or adapted from hmishra2250's github under the MIT liscence. github link: https://github.com/hmishra2250/NTM-One-Shot-TF

def main():
	sess = tf.InteractiveSession()	#create tf session
	#variables/hyperparamaters
	num_reads = 4
	controller_size = 200
	memory_shape = (128, 40)
	num_outputs = 30					#uses five-hot encoding, thus fixed output
	num_classes = 5
	input_size = 20*20
	batch_size = 16						#microbatches of 10
	num_episodes = 10000
	num_batches = num_episodes/batch_size
	num_samples_per_class = 10


	#example batch
	#batch size = 16
	#5 clases, 10 per class
	#input ph = (16, 50, 400)
	#target ph = (16, 50, 30).eval(feed_dict)_decode
	#create placeholders
	input_ph = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, 400))					#batch_size, total examples per ep, image size flattened
	target_ph = tf.placeholder(dtype=tf.float32, shape=(None, num_outputs))			#batch_size, total examples per ep, number of outputs
	inst_ph = tf.placeholder(dtype=tf.int32, shape=(None, 3))

	generator = omniGenerator.OmniglotGenerator(data_folder="./data/omniglot", batch_size=batch_size, num_samples=num_outputs, num_samples_per_class=num_samples_per_class, max_iter=num_batches, num_classes=num_classes)
	output, output_flatten, params = model.MANN(input_ph, target_ph, batch_size=batch_size, num_outputs=num_outputs, memory_shape=memory_shape, controller_size=controller_size, input_size=input_size, num_reads=num_reads, num_samples_per_class=num_samples_per_class, num_classes=num_classes)

	with tf.variable_scope("weights", reuse=True):
			weight_key = tf.get_variable("weight_key", shape=(controller_size, memory_shape[1]))
			bias_key = tf.get_variable("bias_key", shape=memory_shape[1])
			weight_alpha = tf.get_variable("weight_alpha", shape=(controller_size, ((memory_shape[1]+1)*num_reads)))
			bias_alpha = tf.get_variable("bias_alpha", shape=((memory_shape[1]+1)*num_reads))
			weight_sigma = tf.get_variable("weight_sigma", shape=(num_reads, controller_size, 1))
			bias_sigma = tf.get_variable("bias_sigma", shape=(num_reads, 1))
			weight_inputhidden = tf.get_variable("weight_inputhidden", shape=(input_size + num_outputs, 4 * controller_size))
			bias_inputhidden = tf.get_variable("bias_inputhidden", shape=(4 * controller_size))
			weight_output = tf.get_variable("weight_output", shape=(controller_size + num_reads * memory_shape[1], num_outputs))
			bias_output = tf.get_variable("bias_output", shape=(num_outputs))
			weight_readhidden = tf.get_variable("weight_readhidden", shape=(num_reads * memory_shape[1], 4 * controller_size))
			weight_hiddenhidden = tf.get_variable("weight_hiddenhidden", shape=(controller_size, 4 * controller_size))
			gamma = tf.get_variable("gamma", shape=[1], initializer=tf.constant_initializer(0.95))

	params = [weight_key, bias_key, weight_alpha, bias_alpha, weight_sigma, bias_sigma, weight_inputhidden, weight_readhidden, weight_hiddenhidden, bias_inputhidden, weight_output, bias_output]

	#output_target_ph = tf.one_hot(target_ph, depth=generator.num_samples)
	print("Output, target shapes: ", output.get_shape().as_list(), target_ph.get_shape().as_list())
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target_ph), name="cost")
	eps=1e-8				#make sure you don't take log 0
	#cost = -tf.reduce_mean(tf.reduce_sum(tf.stack(tf.split(target_ph, 6, axis=1) * tf.log(tf.split(output+eps, 6, axis=1)))))
	print(target_ph)
	print(output)

	cost = -tf.reduce_mean(tf.reduce_sum(tf.stack(tf.split(tf.reshape(target_ph, shape=(batch_size, num_classes*num_samples_per_class, num_outputs)), 6, axis=2), axis=2) * tf.log(tf.add(tf.reshape(output, shape=(batch_size, num_classes*num_samples_per_class, 6, 5)), eps)), axis=[1, 2, 3]))
	optimizer = tf.train.RMSPropOptimizer(1e-4, decay=0.95, momentum=0.9)		#minimize loss with RMSProp, learning rate, momentum, and decay from DeepMind's MANN paper
	#optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
	train_step = optimizer.minimize(cost, var_list=params)

	#accuracies = utils.accuracy_instance(tf.argmax(output, axis=1), target_ph, batch_size=generator.batch_size)
	#accuracies = utils.accuracy(output_flatten, target_ph, inst_ph, num_classes=num_classes, batch_size=batch_size)
	output_split = tf.split(output, 6, axis=1)
	tmp = tf.Print(output_split, output_split)
	sum_output = tf.stack([tf.one_hot([tf.argmax(t, axis = 1)], depth=1) for t in output_split], axis=1)

	saver = tf.train.Saver()	#create a savepoint of the model
	print("done")

	tf.summary.scalar("cost", cost)
	#for i in range(3):
	#	tf.summary.scalar("accuracy-"+str(i*5), accuracies[i])

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter("/tmp/tboard/newModelFive1/")

	sess.run(tf.global_variables_initializer())
	#saver.restore(sess, "/save/baseNewLoss2.ckpt")

	print("training the model")
	t0 = time.time()

	for i, (batch_input, batch_output, inst) in generator:
		print("b")
		feed_dict = {
			input_ph: batch_input,
			target_ph: batch_output,
			inst_ph: inst
		}

		train_step.run(feed_dict)
		score = cost.eval(feed_dict)
		print(target_ph.eval(feed_dict))
		print(out.eval())

		#acc = accuracies.eval(feed_dict)

		summary = merged.eval(feed_dict)
		train_writer.add_summary(summary, i)
		#accs += acc
		if i % 100 == 0:
			print("batch", i+1, "out of", num_batches, "time", time.time()-t0)
			print("cost", score)
			accuracies = utils.test_f(target_ph.eval(feed_dict), output_flatten.eval(feed_dict))
			print("acc", accuracies)
		'''if i>=0:
			print(accs / 100.0)
			print("Episode ", i, " Accuracy: ", acc, " Loss: ", cost, " Score: ", np.mean(score))
			scores, accs = [], np.zeros(generator.num_samples_per_class)'''

	print("saving the model")
	saver.save(sess, "/save/newModelFive1.ckpt")		#save learned weights and biases
	train_writer.add_graph(sess.graph)		#save graph values (loss, acc)

if __name__ == "__main__":
	main()
