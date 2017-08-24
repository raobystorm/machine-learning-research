import tensorflow as tf
import numpy as np
import os
import time
import librosa
import pickle
import random

n_input = 256 * 64
n_classes = 2
max_iter = 100000
batch_size = 64
random_sample_size = 256
isLoad = False

print('n_input: %d' % n_input)
print('n_classes: %d' % n_classes)
print('max_iter: %d' % max_iter)
print('batch_size: %d' % batch_size)
print('random_sample_size: %d' % random_sample_size)

model_save_path = '/home/centos/audio-recognition/AudioSet/model.ckpt'

data_file = '/home/centos/audio-recognition/AudioSet/data.1503388707'
eval_data_file = '/home/centos/audio-recognition/AudioSet/eval_data.dat'

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
W_conv1 = weight_varible('W_conv1', [11, 11, 1, 48])
b_conv1 = bias_variable('b_conv1', [48])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_wh(h_conv1, 4, 2)

# conv layer-2
W_conv2 = weight_varible('W_conv2', [5, 5, 48, 96])
b_conv2 = bias_variable('b_conv2', [96])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_wh(h_conv2, 4, 2)

# conv layer-3
W_conv3 = weight_varible('W_conv3', [3, 3, 96, 96])
b_conv3 = bias_variable('b_conv3', [96])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool(h_conv3, 3)

# fully-connect-1
W_fc1 = weight_varible('W_fc1', [6 * 6 * 96, 512])
b_fc1 = bias_variable('b_fc1', [512])

h_pool3_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 96])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#dropout-1
keep_prob_1 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_1)

# fully-connect-2
W_fc2 = weight_varible('W_fc2', [512, 64])
b_fc2 = bias_variable('b_fc2', [64])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# dropout-2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_1)

# output layer: softmax
W_fc3 = weight_varible('W_fc3', [64, n_classes])
b_fc3 = bias_variable('b_fc3', [n_classes])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

saver = tf.train.Saver()

#learning_rate
learning_rate = 1e-4

# model training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    data_ = load_data(data_file)
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
        train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob_1: 0.3})
        if i % 5000 == 0:
            test_batch = random_sample(test_data)
            test_accuracy = accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob_1: 1.0})
            print('test accuracy %g' % test_accuracy)
            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy
                print('Model saved in %s' % saver.save(sess, model_save_path))

    test_batch = random_sample(test_data)
    print('test accuracy %g' % test_accuracy)
    if test_accuracy > max_accuracy:
        print('Model saved in %s' % saver.save(sess, model_save_path))
