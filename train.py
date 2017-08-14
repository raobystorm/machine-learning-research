import tensorflow as tf
import numpy as np
import os
import librosa
import pickle
import random

n_input = 128 * 64
n_classes = 2
learning_rate = 0.0001
max_iter = 100000
batch_size = 32
random_sample_size = 128
isLoad = False

model_save_path = '/home/centos/audio-recognition/AudioSet/mdoel.ckpt'

data_file = '/home/centos/audio-recognition/AudioSet/data.dat'
eval_data_file = '/home/centos/audio-recognition/AudioSet/eval_data.dat'

def random_sample(data_batch):
  data_list = []
  label_list = []
  for data in data_batch:
    sample = random.sample(list(data[0]), random_sample_size)
    data_list.append(np.reshape(sample, [n_input]))
    label_list.append(np.reshape(data[1], [n_classes]))
  return data_list, label_list

def get_batch(data, batch_size, iteration):
  start_of_batch = (iteration * batch_size) % len(data)
  end_of_batch = (iteration * batch_size + batch_size) % len(data)
  if start_of_batch < end_of_batch:
    return data[start_of_batch:end_of_batch]
  else:
    start_of_batch = (start_of_batch + 1) % len(data)
    end_of_batch = (end_of_batch + 1) % len(data)
    data_batch = np.vstack((data[start_of_batch:],data[:end_of_batch]))
    return data_batch

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

def weight_varible(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, k):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# paras
W_conv1 = weight_varible([4, 4, 1, 48])
b_conv1 = bias_variable([48])

# conv layer-1
x_image = tf.reshape(x, [-1, 128, 64, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, 2)

# conv layer-2
W_conv2 = weight_varible([3, 3, 48, 96])
b_conv2 = bias_variable([96])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, 2)

# conv layer-3
W_conv3 = weight_varible([3, 3, 96, 160])
b_conv3 = bias_variable([160])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool(h_conv3, 2)

# conv layer-4
W_conv4 = weight_varible([3, 3, 160, 256])
b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool(h_conv4, 2)

# full connection
W_fc1 = weight_varible([8 * 4 * 256, 1024])
b_fc1 = bias_variable([1024])

h_pool4_flat = tf.reshape(h_pool4, [-1, 8 * 4 * 256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([1024, n_classes])
b_fc2 = bias_variable([n_classes])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

saver = tf.train.Saver()

# model training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  data_ = load_data(data_file)
  if isLoad:
    saver.restore(sess, model_save_path)
  else:
    sess.run(tf.global_variables_initializer())
  for i in range(max_iter):
    train_batch = random_sample(get_batch(data_, batch_size, i))
    if i % 100 == 0:
      train_accuacy = accuracy.eval(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuacy))
    train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})
    if i % 10000 == 0:
      test_data = load_data(eval_data_file)
      test_batch = random_sample(test_data)
      print('test accuracy %g' % accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

  print('Model saved in %s' % saver.save(sess, model_save_path))
  test_data = load_data(eval_data_file)
  test_batch = random_sample(test_data)
  print('test accuracy %g' % accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))