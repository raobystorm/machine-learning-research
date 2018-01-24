# Pyhton 2.7

import os
import sys
import cPickle

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')
gallery_ratio = 0.7

import caffe

caffe.set_mode_gpu()
model_def = caffe_root + 'models/mitene_test/alexnet_extraction.prototxt'
model_weights = caffe_root + 'models/mitene_test/full_body.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
net.blobs['data'].reshape(1, 3, 227, 227)

base_folder = '/home/centos/mitene-pre_experiment/results'
test_set = []
reference_set = []

for sub_folder in os.listdir(base_folder):
    extracted_features = []
    img_count = 0
    total_count = len(os.listdir(base_folder))
    reference_count = total_count * gallery_ratio
    for img in os.listdir(base_folder + '/' + sub_folder):
        if os.path.splitext(img)[1] != '.jpg':
            continue
        image = caffe.io.load_image(base_folder + '/' + sub_folder + '/images/' + img)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        if img_count < reference_count:
            reference_set << output
            print 'processed: reference image: ' + img + ', reference count: ' + str(img_count) + '/' + str(reference_count)
        else:
            test_set << output
            print 'processed: test image: ' + img + ', test count: ' + str(img_count - reference_count) + '/' + str(total_count)
        img_count += 1

    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_reference.dat', 'w') as f:
        cPickle.dump(reference_set, f)
    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_test.dat', 'w') as f:
        cPickle.dump(test_set, f)