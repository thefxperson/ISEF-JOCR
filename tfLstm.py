import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

nm_epochs = 6
nm_classes = 10
batch_size = 100
chunk_size = 28
nm_chunks = 28
hdn_units = 200

x = tf.placeholder("float", [None, nm_chunks, chunk_size])
y = tf.placeholder("float")

def lstm(data):
    global x
    hidden_layer = {"weights":tf.Variable(tf.random_normal([hdn_units,nm_classes])),
                    "biases":tf.Variable(tf.random_normal([nm_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x,[-1, chunk_size])
    x = tf.split(x, nm_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(hdn_units,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],hidden_layer["weights"]) + hidden_layer["biases"]
    return output

def train_NN(x):
    prediction = lstm(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(nm_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,nm_chunks,chunk_size))

                i, c = sess.run([optimizer, cost], feed_dict={x:epoch_x,y:epoch_y})

                epoch_loss += c

            print("Epoch ", epoch+1, " completed out of ", nm_epochs, " loss: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("accuracy: ", accuracy.eval({x:mnist.test.images.reshape((-1,nm_chunks,chunk_size)),y:mnist.test.labels}))
        saver = tf.train.Saver()
        saver.save(sess, "./lstmSave.ckpt")

train_NN(x)