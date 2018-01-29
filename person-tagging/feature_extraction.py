# Pyhton 2.7

import os
import sys
import cPickle
import numpy as np

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')
gallery_ratio = 0.95
image_size = (227, 227)

import caffe

class Image(object):
    def __init__(self, image, name):
        self.image = image
        self.name = name


def preprocess_image(image, mean, std):
    image.image -= mean
    image.image /= std
    return image

def feature_extract(image_name, net):
    transformed_image = transformer.preprocess('data', image_name.image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    output = output['fc7'].tolist()
    return [output, image_name.name]


caffe.set_mode_gpu()
model_def = caffe_root + 'models/mitene_test/alexnet_extraction.prototxt'
model_weights = caffe_root + 'models/mitene_test/full_body.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
net.blobs['data'].reshape(1, 3, 227, 227)

base_folder = '/home/centos/mitene-pre_experiment/results'
ref_images_name = []
test_images_name = []
ref_set = []
test_set = []

# Read image data and prepare for preprocess
for sub_folder in os.listdir(base_folder):
    img_count = 1
    img_folder = base_folder + '/' + sub_folder + '/images'
    total_count = len(os.listdir(img_folder))
    reference_count = int(total_count * gallery_ratio)
    for img in os.listdir(img_folder):
        if os.path.splitext(img)[1] != '.jpg':
            continue
        image = caffe.io.load_image(img_folder + '/' + img)
        image = caffe.io.resize_image(image, image_size)
        if img_count <= reference_count:
            ref_images_name.append(Image(image=image, name=img))
        else:
            test_images_name.append(Image(image=image, name=img))
        img_count += 1

    # Preprocess for image, calc mean and deviation
    print 'Start calc image mean and deviation for reference images.' + sub_folder
    ref_image = [ data.image for data in ref_images_name ]
    ref_mean = np.mean(ref_image, axis=0)
    ref_std = np.std(ref_image, axis=0)

    ref_images_name = [ preprocess_image(image, ref_mean, ref_std) for image in ref_images_name ]

    print 'Start calc image mean and deviation for test images.' + sub_folder
    test_image = [ data.image for data in test_images_name ]
    test_mean = np.mean(test_image, axis=0)
    test_std = np.std(test_image, axis=0)

    test_images_name = [ preprocess_image(image, test_mean, test_std) for image in test_images_name ]

    print 'Start feature extraction for reference images.' + sub_folder
    ref_images_name = [ feature_extract(image_name, net) for image_name in ref_images_name ]

    print 'Start feature extraction for test images.' + sub_folder
    test_images_name = [ feature_extract(image_name, net) for image_name in test_images_name ]

    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_reference.dat', 'wb') as f:
        cPickle.dump(ref_images_name, f)

    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_test.dat', 'wb') as f:
        cPickle.dump(test_images_name, f)