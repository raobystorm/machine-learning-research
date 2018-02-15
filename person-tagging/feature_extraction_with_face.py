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
model_weights = caffe_root + 'models/mitene_test/face.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
net.blobs['data'].reshape(1, 3, 227, 227)

base_folder = '/home/centos/mitene-pre_experiment_with_face'


def extract_face_result(data_set, count):
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
        output.append(img)
        res_set.append(output)
    return res_set


def extract_body_results(face_data_set):
    res_set = []
    for face_data in face_data_set:
        data = face_data[0]
        filename = face_data[1].split('/')[-1]
        if os.path.splitext(filename)[1] != '.jpg':
            continue
        filename = filename.split('_')[0] + '_' + filename.split('_')[1] + '.jpg'
        img_path = face_data[1].split('/face_images')[0] + '/body_images/' + filename
        image = caffe.io.load_image(img_path)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output = output['fc7'].tolist()[0]
        data += output
        res_set.append([ data, face_data[1].split('/')[-1] ])
    return res_set


face_ref_set = []
sub_folder = os.listdir(base_folder)[0]
img_folder = base_folder + '/' + sub_folder
face_test_set = [ img_folder + '/face_images/' + img for img in os.listdir(img_folder + '/face_images') ]
for other_folder in os.listdir(base_folder):
    if other_folder == sub_folder:
        continue
    img_folder = base_folder + '/' + other_folder
    face_ref_set += [ img_folder + '/face_images/' + img for img in os.listdir(img_folder + '/face_images') ]

np.random.shuffle(face_test_set)
np.random.shuffle(face_ref_set)
face_test_set = extract_face_result(face_test_set, len(face_test_set))
face_ref_set = extract_face_result(face_ref_set, len(face_ref_set))

model_def = caffe_root + 'models/mitene_test/alexnet_extraction.prototxt'
model_weights = caffe_root + 'models/mitene_test/upper_body.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
net.blobs['data'].reshape(1, 3, 227, 227)

test_res = extract_body_results(face_test_set)
ref_res = extract_body_results(face_ref_set)

with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_reference.dat', 'wb') as f:
    cPickle.dump(ref_res, f)

with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_test.dat', 'wb') as f:
    cPickle.dump(test_res, f)
