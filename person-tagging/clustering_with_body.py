from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import os
import math
from clustering import get_onedir, load_model, compute_facial_encodings
from scipy import misc


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
            face_image_paths = get_onedir(args.input + '/face')
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

            nrof_images = len(image_paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            facial_encodings = compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                embedding_size,nrof_images,nrof_batches,emb_array,args.batch_size,image_paths)
            # sorted_clusters = cluster_facial_encodings(facial_encodings)
            # num_cluster = len(sorted_clusters)
                
            # # Copy image files to cluster folders
            # for idx, cluster in enumerate(sorted_clusters):
            #     #save all the cluster
            #     cluster_dir = join(args.output, str(idx))
            #     if not exists(cluster_dir):
            #         makedirs(cluster_dir)
            #     for path in cluster:
            #         shutil.copy(path, join(cluster_dir, basename(path)))

    body_image_paths = get_onedir(args.input + '/body')
    caffe_root = '../caffe'
    sys.path.insert(0, caffe_root + 'python')

    import caffe
    caffe.set_mode_gpu()
    model_def = caffe_root + 'models/mitene_test/alexnet_extraction.prototxt'
    model_weights = caffe_root + 'models/mitene_test/face.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    


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