#code from snowkylin's github, see readme for citation
#modified to support an output of 30 instead of 25 with five-hot
#excess code that I'm not using (such as ntm) has been removed

from utils import OmniglotDataLoader, five_hot_decode
import tensorflow as tf
import argparse
import numpy as np
from model import NTMOneShotLearningModel
from tensorflow.python import debug as tf_debug
import output_utils
import os
from PIL import Image
from PIL import ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="eval")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--label_type', default="five_hot")
    parser.add_argument('--n_classes', default=14)
    parser.add_argument('--seq_length', default=14)
    parser.add_argument('--augment', default=True)
    parser.add_argument('--model', default="MANN", help='LSTM, MANN, MANN2 or NTM')
    parser.add_argument('--read_head_num', default=4)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--num_epoches', default=1000)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--rnn_size', default=200)
    parser.add_argument('--image_width', default=20)
    parser.add_argument('--image_height', default=20)
    parser.add_argument('--rnn_num_layers', default=1)
    parser.add_argument('--memory_size', default=128)
    parser.add_argument('--memory_vector_dim', default=40)
    parser.add_argument('--shift_range', default=1, help='Only for model=NTM')
    parser.add_argument('--write_head_num', default=1, help='Only for model=NTM. For MANN #(write_head) = #(read_head)')
    parser.add_argument('--test_batch_num', default=1)
    parser.add_argument('--n_train_classes', default=1200)
    parser.add_argument('--n_test_classes', default=423)
    parser.add_argument('--save_dir', default='/save/one_shot_learning')
    parser.add_argument('--tensorboard_dir', default='/summary/one_shot_learning')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == "eval":
    	eval(args)


def train(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(
        image_size=(args.image_width, args.image_height),
        n_train_classses=args.n_train_classes,
        n_test_classes=args.n_test_classes
    )
    with tf.Session() as sess:
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model, sess.graph)
        #print(args)
        print("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tbatch\tloss")
        for b in range(args.num_epoches):

            # Test

            if b % 100 == 0:
                x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                              type='test',
                                                              augment=args.augment,
                                                              label_type="five_hot")
                feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
                output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
                merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)
                # state_list = sess.run(model.state_list, feed_dict=feed_dict)  # For debugging
                # with open('state_long.txt', 'w') as f:
                #     print(state_list, file=f)
                accuracy = test_f(args, y, output)
                for accu in accuracy:
                    print('%.4f' % accu, end='\t')
                print('%d\t%.4f' % (b, learning_loss))

            # Save model

            if b % 5000 == 0 and b > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '/model.ckpt', global_step=b)

            # Train

            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                          type='train',
                                                          augment=args.augment,
                                                          label_type="five_hot")
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            sess.run(model.train_op, feed_dict=feed_dict)

        saver.save(sess, args.save_dir + '/' + args.model + '/model.ckpt', global_step=b)
        train_writer.add_graph(sess.graph)


def test(args):
    model = NTMOneShotLearningModel(args)
    print("model loaded")
    data_loader = OmniglotDataLoader(
        image_size=(args.image_width, args.image_height),
        n_train_classses=args.n_train_classes,
        n_test_classes=args.n_test_classes
    )
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
    print("saver created")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tloss")
        for b in range(args.test_batch_num):
            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                          type='test',
                                                          augment=args.augment,
                                                          label_type="five_hot")
            #print(x_image)
            feed_dict = {model.x_image: x_image, model.x_label: x_label}
            output, state = sess.run([model.o,model.state_list], feed_dict=feed_dict)

        print("saving")
        saver.save(sess, args.save_dir + '/' + args.model + '/memsave/modelTrained.ckpt')
        np.save("chars/test.npy", state[-1])
        print("saved")

def eval(args):
    print("eval")
    model = NTMOneShotLearningModel(args)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model + "/memsave")
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored")
        #load images from anki
        imgs = []
        x = []
        for direc, subdir, file in os.walk("evaluation/"):
            imgs.append([Image.open(direc + filename).copy() for filename in file])

        for i in range(len(imgs[0])):
            image = ImageOps.invert(imgs[0][i].convert('L')).resize((20,20))
            np_image = np.reshape(np.array(image, dtype=np.float32),newshape=(20 * 20))
            max_value = np.max(np_image)    # normalization is important
            if max_value > 0.:
                np_image = np_image / max_value
            x.append(np_image)
        x = [x, x]
        #create predefined target list, xinput shifted
        fooDict = np.load("chars/kanji_list.npy").item()
        xlab = [fooDict["あ"], fooDict["お"], fooDict["か"],fooDict["シ"],fooDict["ソ"],fooDict["ツ"],fooDict["ン"],fooDict["日"],fooDict["月"],fooDict["目"],fooDict["耳"],fooDict["人"],fooDict["火"],fooDict["入"]]
        keys = ["あ","お","か","シ","ソ","ツ","ン","day", "month", "eye", "ear", "person", '"fire"', "enter"]
        xlab = xlab[1:] + [xlab[0]]
        xlab = [xlab, xlab]
        #get outputs
        feed_dict = {model.x_image: x, model.x_label: xlab}
        out = sess.run(model.o, feed_dict=feed_dict)
        #get top 3 predictions from jisho, do the thing with the output, and print
        outDict = np.load("chars/output_list.npy").item()
        for i in range(out.shape[1]):
            print(i)
            for j in range(1,4):
                key = output_utils.combineOut(out[0,i,:], output_utils.getContext(keys[i], 75, j))
                #key = output_utils.combineOut(output_utils.getContext(keys[i], 1, j), output_utils.getContext(keys[i], 1, j))

                if key in outDict:
                    print(outDict[key])
                else:
                    print(key)


def test_f(args, y, output):
    correct = [0] * args.seq_length
    total = [0] * args.seq_length
    y_decode = five_hot_decode(y)
    output_decode = five_hot_decode(output)
    for i in range(np.shape(y)[0]):
        y_i = y_decode[i]
        output_i = output_decode[i]
        # print(y_i)
        # print(output_i)
        class_count = {}
        for j in range(args.seq_length):
            if y_i[j] not in class_count:
                class_count[y_i[j]] = 0
            class_count[y_i[j]] += 1
            total[class_count[y_i[j]]] += 1
            if y_i[j] == output_i[j]:
                correct[class_count[y_i[j]]] += 1
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 11)]


if __name__ == '__main__':
    main()