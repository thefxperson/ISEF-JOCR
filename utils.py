#code from snowkylin's github, see readme for citation
#modified to support an output of 30 instead of 25 with five-hot
#excess code that I'm not using (such as ntm) has been removed

import numpy as np
import os
from PIL import Image
from PIL import ImageOps
# from skimage import io
# from skimage import transform
# from skimage import util


def generate_random_strings(batch_size, seq_length, vector_dim):
    return np.random.randint(0, 2, size=[batch_size, seq_length, vector_dim]).astype(np.float32)

def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim, ), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


def five_hot_decode(x):
    x = np.reshape(x, newshape=np.shape(x)[:-1] + (6, 5))
    def f(a):
        return sum([a[i] * 5 ** i for i in range(6)])
    foo = np.apply_along_axis(f, -1, np.argmax(x, axis=-1))
    return foo


def baseN(num,b):
    return ((num == 0) and  "0" ) or ( baseN(num // b, b).lstrip("0") + "0123456789abcdefghijklmnopqrstuvwxyz"[num % b])


class OmniglotDataLoader:
    def __init__(self, data_dir='./data', image_size=(20, 20), n_train_classses=1200, n_test_classes=423):
        self.data = []
        self.image_size = image_size
        for dirname, subdirname, filelist in os.walk(data_dir):
            if filelist:
                self.data.append(
                    [Image.open(dirname + '/' + filename).copy() for filename in filelist]
                )

        self.train_data = self.data[:n_train_classses]
        #self.test_data = self.data[-n_test_classes:]
        self.test_data = self.data[:]
        #print(self.data)

    def fetch_batch(self, n_classes, batch_size, seq_length,
                    type='test',
                    sample_strategy='uniform',
                    augment=True,
                    label_type='five_hot',
                    myset=True):
        if type == 'train':
            data = self.train_data
        elif type == 'test':
            data = self.test_data
        if myset:
            classes = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                       [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
                       [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44],
                       [45,46,47,48,49,50,51,52,53,54,55,56,57,58,59],
                       [60,61,62,63,64,65,66,67,68,69,70,71,72,73,74],
                       [75,76,77,78,79,80,81,82,83,84,85,86,87,88,89],
                       [90,91,92,93,94,95,96,97,98,0,1,2,3,4,5]]
        #classes = [np.random.choice(range(len(data)), replace=False, size=n_classes) for _ in range(batch_size)]
        #load the data based on batch
        if sample_strategy == 'random':         # #(sample) per class may not be equal (sec 7)
            seq = np.random.randint(0, n_classes, [batch_size, seq_length])
        elif sample_strategy == 'uniform':      # #(sample) per class are equal
            seq = np.array([np.concatenate([[j] * int(seq_length / n_classes) for j in range(n_classes)])
                   for i in range(batch_size)])
            #for i in range(batch_size):
            #    np.random.shuffle(seq[i, :])
        #self.rand_rotate_init(n_classes, batch_size)
        seq_pic = [[self.augment(data[classes[i][j]][np.random.randint(0, len(data[classes[i][j]]))],
                                 batch=i, c=j,
                                 only_resize=True)
                   for j in seq[i, :]]
                   for i in range(batch_size)]
        #print(seq)
        #label_dict = [[[int(j) for j in list(baseN(i, 5)) + [0] * (6 - len(baseN(i, 5)))] for i in np.random.choice(range(5 ** 6), replace=False, size=n_classes)] for _ in range(batch_size)]
        #seq_encoded_ = np.array([[label_dict[b][i] for i in seq[b]] for b in range(batch_size)])
        #seq_encoded = np.reshape(one_hot_encode(seq_encoded_, dim=5), newshape=[batch_size, seq_length, -1])
        
        #seq_pic = [[data[classes[i][j]] for j in range(15)] for i in range(batch_size)]
        labs = np.load("chars/kanji_list.npy").item()
        alp_string = np.load("chars/output_list.npy").item()
        seq_encoded =[]
        foob = []
        for i in range(batch_size):
            for j in seq[i,:]:
                if j <= 46:
                    foob.append(labs[alp_string[self.alphaHot(j+2136)]])
                elif j < 91:
                    t = self.alphaHot(j+2168)
                    #print(t)
                    #print(alp_string[t])
                    foob.append(labs[alp_string[t]])
                elif j == 91:
                    foob.append(labs[alp_string[self.alphaHot(1582)]])
                elif j == 92:
                    foob.append(labs[alp_string[self.alphaHot(512)]])
                elif j == 93:
                    foob.append(labs[alp_string[self.alphaHot(1946)]])
                elif j == 94:
                    foob.append(labs[alp_string[self.alphaHot(809)]])
                elif j == 95:
                    foob.append(labs[alp_string[self.alphaHot(1069)]])
                elif j == 96:
                    foob.append(labs[alp_string[self.alphaHot(132)]])
                elif j == 97:
                    foob.append(labs[alp_string[self.alphaHot(1583)]])
            seq_encoded.append(foob)
            foob = []

        seq_pic_arr = np.array(seq_pic)
        seq_encoded_shifted = seq_encoded[1:] + [seq_encoded[0]]#np.concatenate([np.zeros(shape=[batch_size, 1, 30]), seq_encoded[:, :-1, :]], axis=1)
        seq_encoded_arr = np.array(seq_encoded)
        seq_encshift_arr = np.array(seq_encoded_shifted)
        return seq_pic_arr, seq_encshift_arr, seq_encoded_arr

    def rand_rotate_init(self, n_classes, batch_size):
        self.rand_rotate_map = np.random.randint(0, 4, [batch_size, n_classes])

    def augment(self, image, batch, c, only_resize=False):
        if only_resize:
            image = ImageOps.invert(image.convert('L')).resize(self.image_size)
        else:
            rand_rotate = self.rand_rotate_map[batch, c] * 90                       # rotate by 0, pi/2, pi, 3pi/2
            image = ImageOps.invert(image.convert('L')) \
                .rotate(rand_rotate + np.random.rand() * 22.5 - 11.25,
                        translate=np.random.randint(-10, 11, size=2).tolist()) \
                .resize(self.image_size)   # rotate between -pi/16 to pi/16, translate bewteen -10 and 10
        np_image = np.reshape(np.array(image, dtype=np.float32),
                          newshape=(self.image_size[0] * self.image_size[1]))
        max_value = np.max(np_image)    # normalization is important
        if max_value > 0.:
            np_image = np_image / max_value
        return np_image

    def alphaHot(self, num):
        baseFive = self.changeBase(num)        #changeBase returns an array, thus is called 1x per loop to get new array
        foo = ""
        for j in range(6):
            foo += chr(97+baseFive[j])                  #change the base from 10 to 5 so it can be encoded using letters a->e
        return foo

    def changeBase(self, number):   #changes a number in base 10 to a number in base 5
        remainder = []
        for i in range(6):
            remainder.append(number % 5)    #mod func returns remainder
            number = int(number / 5)
        return remainder[::-1]          #reverse nums

#foo = OmniglotDataLoader()
#a, b, c = foo.fetch_batch(15, 7, 150)
#print(b.shape)