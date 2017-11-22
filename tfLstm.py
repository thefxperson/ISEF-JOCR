import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

nm_epochs = 6		#number of epochs
nm_classes = 10		#number of classes to classify
batch_size = 100	#number of images per batch
chunk_size = 28		#image x size
nm_chunks = 28		#image y size
hdn_units = 200		#number of hidden units

#placeholder variables for images and labels
x = tf.placeholder("float", [None, nm_chunks, chunk_size])
y = tf.placeholder("float")

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
        optimizer = tf.train.AdamOptimizer().minimize(cost)										#minimize loss with Adam

        tf.summary.scalar("cost", cost)

    saver = tf.train.Saver()	#create a savepoint of the model

    #begin the training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())	#init the variables
        merged = tf.summary.merge_all()
        global summary_val

        #run through all epochs
        for epoch in range(nm_epochs):
            epoch_loss = 0	#loss in current epoch
            for i in range(int(mnist.train.num_examples/batch_size)):		#for each batch in the data
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)		#train the machine (forward prop?)
                epoch_x = epoch_x.reshape((batch_size,nm_chunks,chunk_size))		#reshape results

                i, c, summary_val = sess.run([optimizer, cost, merged], feed_dict={x:epoch_x,y:epoch_y})	#backpropigate using optimizer, update weights,ect

                epoch_loss += c 		#cost for epoch

            print("Epoch ", epoch+1, " completed out of ", nm_epochs, " loss: ", epoch_loss)

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

train_NN(x)
'''lstm(x)	#create variables before loading
with tf.Session() as sess:
    saver = tf.train.Saver()	#load learned weights+biases
    saver.restore(sess, "./lstmSave.ckpt")'''