# Pyhton 2.7

import os
import sys

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_gpu()
model_def = caffe_root + 'models/mitene_test/alexnet_extraction.prototxt'
model_weights = caffe_root + 'models/mitene_test/full_body.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
net.blobs['data'].reshape(1, 3, 227, 227)

base_folder = '/home/centos/mitene-pre_experiment/results'
json_str = '['
for sub_folder in os.listdir(base_folder):
    extracted_features = []
    for img in os.listdir(base_folder + '/' + sub_folder):
        if os.path.splitext(img)[1] != '.jpg':
            continue
        image = caffe.io.load_image(base_folder + '/' + sub_folder + '/' + img)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        json_str += '{"' + img + '" : ' + str(output['fc7'].tolist()[0]) + '},\n'
        print "processed: " + img

    json_str = json_str[:-1][:-1]
    json_str += ']'
    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '.dat', 'w') as f:
        f.write(json_str)