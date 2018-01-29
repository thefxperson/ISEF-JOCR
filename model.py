#code from snowkylin's github, see readme for citation
#modified to support an output of 30 instead of 25 with five-hot
#excess code that I'm not using (such as ntm) has been removed

import tensorflow as tf
import numpy as np


class NTMOneShotLearningModel():
    def __init__(self, args):
        args.output_dim = 30

        self.x_image = tf.placeholder(dtype=tf.float32,
                                      shape=[args.batch_size, args.seq_length, args.image_width * args.image_height])
        self.x_label = tf.placeholder(dtype=tf.float32,
                                      shape=[args.batch_size, args.seq_length, args.output_dim])
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[args.batch_size, args.seq_length, args.output_dim])

        import ntm.mann_cell as mann_cell
        cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                    head_num=args.read_head_num)

        state = cell.zero_state(args.batch_size, tf.float32)
        self.state_list = [state]   # For debugging
        self.o = []
        for t in range(args.seq_length):
            output, state = cell(tf.concat([self.x_image[:, t, :], self.x_label[:, t, :]], axis=1), state)
            # output, state = cell(self.y[:, t, :], state)
            with tf.variable_scope("o2o", reuse=(t > 0)):
                o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                o2o_b = tf.get_variable('o2o_b', [args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)

            output = tf.stack([tf.nn.softmax(o) for o in tf.split(output, 6, axis=1)], axis=1)
            self.o.append(output)
            self.state_list.append(state)
        self.o = tf.stack(self.o, axis=1)
        self.state_list.append(state)

        eps = 1e-8
        self.learning_loss = -tf.reduce_mean(tf.reduce_sum(tf.stack(tf.split(self.y, 6, axis=2), axis=2) * tf.log(self.o + eps), axis=[1, 2, 3]))

        self.o = tf.reshape(self.o, shape=[args.batch_size, args.seq_length, -1])
        self.learning_loss_summary = tf.summary.scalar('learning_loss', self.learning_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, momentum=0.9, decay=0.95)
            # gvs = self.optimizer.compute_gradients(self.learning_loss)
            # capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            # self.train_op = self.optimizer.apply_gradients(gvs)
            self.train_op = self.optimizer.minimize(self.learning_loss)