from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import importlib
import argparse
import sys
import facenet
import os
import math
from clustering import get_onedir, load_model, compute_facial_encodings, _chinese_whispers, cluster_facial_encodings
from scipy import misc
import pdb

body_image_size = (227, 227)

def compute_body_encodings(image_size, paths):
    """ Compute Facial Encodings

        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.

        Inputs:
            image_paths: a list of image paths

        Outputs:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

    """
    body_encodings = {}
    os.chdir("/home/centos/caffe")
    caffe_root = './'
    sys.path.insert(0, caffe_root + 'python')

    import caffe
    caffe.set_mode_gpu()
    model_def = caffe_root + 'models/mitene_test/alexnet_extraction.prototxt'
    model_weights = caffe_root + 'models/mitene_test/upper_body.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    net.blobs['data'].reshape(1, 3, body_image_size[0], body_image_size[1])

    for i, path in enumerate(paths):
        image = caffe.io.load_image(path)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        transformed_image = facenet.prewhiten(transformed_image)
        output = net.forward()
        output = output['fc7'].tolist()
        res = np.reshape(np.array(output), (4096,))
        rmin = np.min(res)
        rmax = np.max(res)
        res = (res - rmin) / (rmax - rmin)
        body_encodings[path] = res

    return body_encodings


def main(args):
    """ Main

    Given a list of images, save out facial encoding data files and copy
    images into folders of face clusters.

    """
    from os.path import join, basename, exists
    from os import makedirs
    import numpy as np
    import shutil
    import sys

    if not exists(args.output):
        makedirs(args.output)

    facial_encodings = {}

    with tf.Graph().as_default():
        with tf.Session() as sess:
            face_image_paths = get_onedir(str(args.input) + '/face')
            #image_list, label_list = facenet.get_image_paths_and_labels(train_set)

            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            load_model(args.model_dir, meta_file, ckpt_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            print("image_size:",image_size)
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on images') 

            nrof_images = len(face_image_paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            facial_encodings = compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                embedding_size,nrof_images,nrof_batches,emb_array,args.batch_size,face_image_paths)
            # sorted_clusters = cluster_facial_encodings(facial_encodings)
            # num_cluster = len(sorted_clusters)

            # Copy image files to cluster folders
            # for idx, cluster in enumerate(sorted_clusters):
            #     #save all the cluster
            #     cluster_dir = join(args.output, str(idx))
            #     if not exists(cluster_dir):
            #         makedirs(cluster_dir)
            #     for path in cluster:
            #         shutil.copy(path, join(cluster_dir, basename(path)))

    body_image_paths = get_onedir(args.input + '/body')
    emb_array = np.zeros((len(body_image_paths), 4096))
    body_encodings = compute_body_encodings(body_image_size, body_image_paths)
    sorted_facial_clusters = _chinese_whispers(facial_encodings.items(), threshold=0.92)

    sorted_body_clusters = _chinese_whispers(body_encodings.items(), threshold=45.0)
    for idx, cluster in enumerate(sorted_body_clusters):
        #save all the cluster
        cluster_dir = join(args.output, str(idx))
        if not exists(cluster_dir):
            makedirs(cluster_dir)
        for path in cluster:
            shutil.copy(path, join(cluster_dir, basename(path)))


def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--model_dir', type=str, help='model dir', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', required=30)
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main(parse_args())