# Pyhton 2.7

import os
import sys
import cPickle
import numpy as np

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')
ref_count = 1000
test_count = 200

import caffe

caffe.set_mode_gpu()
model_def = caffe_root + 'models/mitene_test/alexnet_extraction.prototxt'
model_weights = caffe_root + 'models/mitene_test/full_body.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
net.blobs['data'].reshape(1, 3, 227, 227)

base_folder = '/home/centos/mitene-pre_experiment/results'


def extract_result(data_set, count):
    res_set = []
    for img in data_set:
        if os.path.splitext(img)[1] != '.jpg':
            continue
        if len(res_set) >= count:
            return res_set
        image = caffe.io.load_image(img)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output = output['fc7'].tolist()
        output.append(img.split('/')[-1])
        res_set.append(output)
    return res_set


for sub_folder in os.listdir(base_folder):
    ref_set = []
    img_count = 1
    img_folder = base_folder + '/' + sub_folder + '/images'
    test_set = [ img_folder + '/' + img for img in os.listdir(img_folder) ]
    for other_folder in os.listdir(base_folder):
        if other_folder == sub_folder:
            continue
        img_folder = base_folder + '/' + other_folder + '/images'
        ref_set +=  [ img_folder + '/' + img for img in os.listdir(img_folder) ]

    np.random.shuffle(test_set)
    np.random.shuffle(ref_set)
    test_set = test_set[:test_count]
    ref_set = ref_set[:ref_count]

    test_res = extract_result(test_set, test_count)
    ref_res = extract_result(ref_set, ref_count)

    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_reference.dat', 'wb') as f:
        cPickle.dump(ref_res, f)

    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_test.dat', 'wb') as f:
        cPickle.dump(test_res, f)