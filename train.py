import tensorflow as tf
import numpy as np
import os
import time
import librosa
import dill
import pickle
import glob
import random

n_input = 256 * 64
n_classes = 2
max_iter = 200000
batch_size = 64
random_sample_size = 256
isLoad = False

print('n_input: %d' % n_input)
print('n_classes: %d' % n_classes)
print('max_iter: %d' % max_iter)
print('batch_size: %d' % batch_size)
print('random_sample_size: %d' % random_sample_size)

model_save_path = 'AudioSet/model.ckpt'

data_files_path = 'AudioSet/'
eval_data_file = 'AudioSet/eval_data.dat'
#eval_data_file = 'test/test_medium/eval.prod.dat'

def random_sample(data_batch):
    data_list = []
    label_list = []
    random.seed(int(time.time()))
    for data in data_batch:
        start_idx = random.randint(0, len(data[0]) - random_sample_size)
        sample = data[0][start_idx : start_idx + random_sample_size]
        data_list.append(np.reshape(sample, [n_input]))
        label_list.append(np.reshape(data[1], [n_classes]))
    return data_list, label_list

def get_batch(data, batch_size, iteration):
    start_of_batch = (iteration * batch_size) % len(data)
    end_of_batch = (iteration * batch_size + batch_size) % len(data)
    if start_of_batch < end_of_batch:
        return data[start_of_batch:end_of_batch]
    elif start_of_batch is not len(data) and end_of_batch is not 0:
        data_batch = np.vstack((data[start_of_batch:],data[:end_of_batch]))
        return data_batch
    elif start_of_batch is len(data):
        return data[0:batch_size]
    else:
        return data[(len(data)-batch_size):len(data)]

def load_data(file):
    with open(file, 'rb') as fp:
        data_block = pickle.load(fp)
        for data in data_block:
            data[0] = data[0].astype(np.float32)
            data[1] = np.asarray(data[1])
            data[1] = data[1].astype(np.float32)
        return data_block

def load_data_files_from_path(path):
    files = glob.glob(data_files_path + '/data.clip*')
    data_block = []
    for f in files:
        with open(f, 'rb') as fp:
            data = dill.load(fp)
            data_block.extend(data)
    for data in data_block:
        data[0] = data[0].astype(np.float32)
        data[1] = np.asarray(data[1])
        data[1] = data[1].astype(np.float32)
    return data_block

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, n_input])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

def weight_varible(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

def bias_variable(name, shape):
    return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape, dtype=tf.float32))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def max_pool_wh(x, w, h):
    return tf.nn.max_pool(x, ksize=[1, w, h, 1], strides=[1, w, h, 1], padding='SAME')

# Reshape input
x_image = tf.reshape(x, [-1, 256, 64, 1])

# conv layer-1
W_conv1 = weight_varible('W_conv1', [3, 3, 1, 64])
b_conv1 = bias_variable('b_conv1', [64])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# conv layer-2
W_conv2 = weight_varible('W_conv2', [3, 3, 64, 64])
b_conv2 = bias_variable('b_conv2', [64])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_wh(h_conv2, 2, 1)

# conv layer-3
W_conv3 = weight_varible('W_conv3', [3, 3, 64, 128])
b_conv3 = bias_variable('b_conv3', [128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# conv layer-4
W_conv4 = weight_varible('W_conv4', [3, 3, 128, 128])
b_conv4 = bias_variable('b_conv4', [128])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)]
h_pool4 = max_pool_wh(h_conv4, 2, 1)

# conv layer-5
W_conv5 = weight_varible('W_conv5', [3, 3, 256, 256])
b_conv5 = bias_variable('b_conv5', [256])

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

# conv layer-6
W_conv6 = weight_varible('W_conv6', [3, 3, 256, 256])
b_conv6 = bias_variable('b_conv6', [256])

h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

# conv layer-7
W_conv7 = weight_varible('W_conv7', [3, 3, 256, 256])
b_conv7 = bias_variable('b_conv7', [256])

h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)
h_pool7 = max_pool(h_conv7, 2)

# conv layer-8
W_conv8 = weight_varible('W_conv8', [3, 3, 512, 512])
b_conv8 = bias_variable('b_conv8', [512])

h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8) + b_conv8)

# conv layer-9
W_conv9 = weight_varible('W_conv9', [3, 3, 512, 512])
b_conv9 = bias_variable('b_conv9', [512])

h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)

# conv layer-10
W_conv10 = weight_varible('W_conv10', [3, 3, 512, 512])
b_conv10 = bias_variable('b_conv10', [512])

h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10)
h_pool10 = max_pool(h_conv10, 2)

# conv layer-11
W_conv11 = weight_varible('W_conv11', [3, 3, 512, 512])
b_conv11 = bias_variable('b_conv11', [512])

h_conv11 = tf.nn.relu(conv2d(h_pool10, W_conv11) + b_conv11)

# conv layer-12
W_conv12 = weight_varible('W_conv12', [3, 3, 512, 512])
b_conv12 = bias_variable('b_conv12', [512])

h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)

# conv layer-13
W_conv13 = weight_varible('W_conv13', [3, 3, 512, 512])
b_conv13 = bias_variable('b_conv13', [512])

h_conv13 = tf.nn.relu(conv2d(h_conv12, W_conv13) + b_conv13)
h_pool13 = max_pool(h_conv13, 2)

# fully-connect-1
W_fc1 = weight_varible('W_fc1', [8 * 8 * 512, 4096])
b_fc1 = bias_variable('b_fc1', [4096])

h_pool13_flat = tf.reshape(h_pool13, [-1, 8 * 8 * 512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool13_flat, W_fc1) + b_fc1)

#dropout-1
keep_prob_1 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_1)

# fully-connect-2
W_fc2 = weight_varible('W_fc2', [4096, 4096])
b_fc2 = bias_variable('b_fc2', [4096])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# dropout-2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_1)

# output layer: softmax
W_fc3 = weight_varible('W_fc3', [4096, n_classes])
b_fc3 = bias_variable('b_fc3', [n_classes])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

saver = tf.train.Saver()

#learning_rate
learning_rate = 1e-3

# model training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    data_ = load_data_files_from_path(data_files_path)
    random.shuffle(data_)
    test_data = load_data(eval_data_file)
    max_accuracy = 0.0

    if isLoad:
        saver.restore(sess, model_save_path)
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(max_iter):
        train_batch = random_sample(get_batch(data_, batch_size, i))
        if i % 800 == 0:
            train_accuacy = accuracy.eval(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob_1: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuacy))
        train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob_1: 0.5})
        if i % 5000 == 0:
            test_batch = random_sample(test_data)
            test_accuracy = accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob_1: 1.0})
            print('test accuracy %g' % test_accuracy)
            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy
                print('Model saved in %s' % saver.save(sess, model_save_path))

    test_batch = random_sample(test_data)
    test_accuracy = accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob_1: 1.0})
    print('test accuracy %g' % test_accuracy)
    if test_accuracy > max_accuracy:
        print('Model saved in %s' % saver.save(sess, model_save_path))
