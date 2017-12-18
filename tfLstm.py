import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import imageManager

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

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

'''lstm(x)	#create variables before loading
with tf.Session() as sess:
    saver = tf.train.Saver()	#load learned weights+biases
    saver.restore(sess, "./lstmSave.ckpt")'''