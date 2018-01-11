import tensorflow as tf
import numpy as np

#the following code has been taken or adapted from hmishra2250's github under the MIT liscence. github link: https://github.com/hmishra2250/NTM-One-Shot-TF

def cosine_similarity(x, y, eps=1e-6):
	z = tf.matmul(x, tf.transpose(y, perm=[0,2,1]))
	z /= tf.sqrt(tf.multiply(tf.expand_dims(tf.reduce_sum(tf.multiply(x,x), 2), 2), tf.expand_dims(tf.reduce_sum(tf.multiply(y,y), 2), 1)) + eps)
	return z

def shared_float32(x, name=""):
	return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), tf.float32), name=name)

def update_tensor(V, dim2, val):
	val = tf.cast(val, V.dtype)

	def body(foo, body_args):
		v, d2, chg = body_args

		d2_int = tf.cast(d2, tf.int32)
		if len(chg.get_shape().as_list()) == 0:
			chg = [chg]
		else:
			chg = tf.reshape(chg, shape=[1]+chg.get_shape().as_list())
		oob = lambda : tf.slice(tf.concat([v[:d2_int], chg], axis=0), tf.range(0, len(v.get_shape().as_list())), v.get_shape().as_list())
		inb = lambda : tf.slice(tf.concat([v[:d2_int], chg, v[d2_int + 1:]], axis=0), tf.constant(0, shape=[len(v.get_shape().as_list())]), v.get_shape().as_list())
		return tf.cond(tf.less(d2_int + 1, v.get_shape().as_list()[0]), inb, oob)

	Z = tf.scan(body, elems=(V, dim2, val), initializer=tf.constant(1, shape=V.get_shape().as_list()[1:], dtype=tf.float32), name="Scan_Update")
	return Z

def shared_glorot_uniform(shape, dtype=tf.float32, name='', n=None):
	if isinstance(shape,int):
		high = np.sqrt(6. / shape)
		shape = [shape]
	else:
		high = np.sqrt(6. / (np.sum(shape[:2]) * np.prod(shape[2:])))
	shape = shape if n is None else [n] + list(shape)
	return tf.Variable(tf.random_uniform(shape, minval=-high, maxval=high, dtype=dtype, name=name))
	
def shared_zeros(shape, dtype=tf.float32, name='', n=None):
	shape = shape if n is None else (n,) + tuple(shape)
	return tf.Variable(tf.zeros(shape, dtype=dtype), name=name)
	
def shared_one_hot(shape, dtype=tf.float32, name='', n=None):
	shape = (shape,) if isinstance(shape,int) else shape
	shape = shape if n is None else (n,) + shape
	initial_vector = np.zeros(shape, dtype=np.float32)
	initial_vector[...,0] = 1
	return tf.Variable(tf.cast(initial_vector, tf.float32), name=name)
	
def weight_and_bias_init(shape, dtype=tf.float32, name='', n=None):
	return (shared_glorot_uniform(shape, dtype=dtype, name='W_' + name, n=n), shared_zeros((shape[1],), dtype=dtype, name='b_' + name, n=n))

def accuracy_instance(predictions, targets, n=[1, 2, 3, 4, 5, 10], nb_classes=5, nb_samples_per_class=10, batch_size=1, num_outputs=30):
	targets = tf.cast(targets, predictions.dtype)

	accuracy = tf.constant(value=0, shape=(batch_size, nb_samples_per_class), dtype=tf.float32)
	indices = tf.constant(value=0, shape=(batch_size, num_outputs+1), dtype=tf.float32)

	def step_(step_arg1, step_arg2):
		accuracy, indices = step_arg1
		p, t = step_arg2

		p = tf.cast(p, tf.int32)
		t = tf.cast(t, tf.int32)
		##Accuracy Update
		batch_range = tf.cast(tf.range(0, batch_size), dtype=tf.int32)
		gather = tf.cast(tf.gather_nd(indices,tf.stack([tf.range(0,p.get_shape().as_list()[0]), t], axis=1)), tf.int32)
		index = tf.cast(tf.stack([batch_range, gather], axis=1), dtype=tf.int64)
		val = tf.cast(tf.equal(p, t), tf.float32)
		delta = tf.SparseTensor(indices=index, values=val, dense_shape=tf.cast(accuracy.get_shape().as_list(), tf.int64))
		accuracy = accuracy + tf.sparse_tensor_to_dense(delta)
		##Index Update
		index = tf.cast(tf.stack([batch_range, t], axis=1), dtype=tf.int64)
		val = tf.constant(1.0, shape=[batch_size])
		delta = tf.SparseTensor(indices=index, values=val, dense_shape=tf.cast(indices.get_shape().as_list(), dtype=tf.int64))
		indices = indices + tf.sparse_tensor_to_dense(delta)
		return [accuracy, indices]

	print(predictions)
	print(targets)
	accuracy, indices = tf.scan(step_, elems=(tf.transpose(predictions, perm=[1, 0]), tf.transpose(targets, perm=[1, 0])),initializer=[accuracy, indices], name="Scan_Metric_Last")

	accuracy = accuracy[-1]

	accuracy = tf.reduce_mean(accuracy / nb_classes , axis=0)

	return accuracy

	