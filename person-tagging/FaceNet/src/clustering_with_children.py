from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import facenet
import os
import math
import concurrent.futures
import boto3
import zipfile
import shutil
from chameleon import cluster_encodings
def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    #return 1/np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return np.sum(face_encodings*face_to_compare,axis=1)


def _chinese_whispers(encoding_list, threshold=0.92, iterations=20):
    """ Chinese Whispers Algorithm

    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    #from face_recognition.api import _face_distance
    from random import shuffle
    import networkx as nx
    # Create graph
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(list(cluster_nodes))
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            #use the max sum of neighbor weights class as current node's class
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters


def cluster_facial_encodings(facial_encodings):
    """ Cluster facial encodings

        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.

        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest

    """

    if len(facial_encodings) <= 1:
        print ("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings.items())
    return sorted_clusters


def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))


def compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                    embedding_size,nrof_images,nrof_batches,emb_array,batch_size,paths):
    """ Compute Facial Encodings

        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.

        Inputs:
            image_paths: a list of image paths

        Outputs:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

    """

    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

    facial_encodings = {}
    for x in range(nrof_images):
        facial_encodings[paths[x]] = emb_array[x,:]

    return facial_encodings


def get_onedir(paths):
    dataset = []
    path_exp = os.path.expanduser(paths)
    if os.path.isdir(path_exp):
        images = os.listdir(path_exp)
        image_paths = [os.path.join(path_exp,img) for img in images]

        for x in image_paths:
            if os.path.getsize(x)>0:
                dataset.append(x)

    return dataset


def zip_and_upload(sorted_clusters, fam_id):
    filename = os.path.join('/tmp/', fam_id + '.zip')
    with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted_clusters[0]:
            write_path = os.path.join(fam_id, os.path.basename(path))
            zf.write(filename=path, arcname=write_path)

    s3 = boto3.session.Session().resource('s3')
    s3.Bucket('mitene-deeplearning-dataset').upload_file(filename, 'faces/' + os.path.basename(filename))
    return fam_id


def compute_cluster_center(cluster, face_encodings):
    cluster_paths = filter(lambda x: x in cluster, face_encodings.keys())
    cluster_encodings = [face_encodings[x] for x in cluster_paths]
    return np.mean(cluster_encodings, axis=0)


def data_cleaning(sorted_clusters, face_encodings):
    first_cluster_center = compute_cluster_center(sorted_clusters[0], face_encodings)

    for path in sorted_clusters[0]:
        uuid = os.path.basename(path).split('_')[0]
        paths_with_uuid = list(filter(lambda x: uuid in x, sorted_clusters[0]))
        if len(paths_with_uuid) > 1:
            max_dis = 0
            for path in paths_with_uuid:
                distance = np.sum(face_encodings[path]*first_cluster_center)
                if distance > max_dis:
                    max_dis = distance
                    max_path = path

            for path in paths_with_uuid:
                if path != max_path:
                    sorted_clusters[0].remove(path)

    return sorted_clusters


def sort_dirs(fam_input, jobs):
    jobs = list(filter(lambda x: os.path.isdir(os.path.join(fam_input, x)), jobs))
    jobs = list(map(lambda x: int(x), jobs))
    jobs.sort()
    return list(map(lambda x: str(x), jobs))


def save_one_cluster(output, cluster):
    cluster_dir = os.path.join(output)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    for path in cluster:
        shutil.copy(path, os.path.join(cluster_dir, os.path.basename(path)))
    return output


def run(fam_input, fam_output, children):
    """ Main

    Given a list of images, save out facial encoding data files and copy
    images into folders of face clusters.

    """
    import numpy as np

    batch_size = 500
    model_dir = '/home/ubuntu/FaceNet/20170512-110547'

    if os.path.exists(fam_output):
        shutil.rmtree(fam_output)

    os.mkdir(fam_output)

    with tf.Graph().as_default():
        with tf.Session() as sess:

            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            load_model(model_dir, meta_file, ckpt_file)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                cluster_centers = []
                for job in sort_dirs(fam_input, os.listdir(fam_input)):
                    futures = []
                    input_dir = os.path.join(fam_input, job)
                    image_paths = get_onedir(input_dir)
                    # image_list, label_list = facenet.get_image_paths_and_labels(train_set)

                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    image_size = images_placeholder.get_shape()[1]
                    embedding_size = embeddings.get_shape()[1]

                    nrof_images = len(image_paths)
                    nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
                    emb_array = np.zeros((nrof_images, embedding_size))
                    facial_encodings = compute_facial_encodings(sess, images_placeholder, embeddings, phase_train_placeholder,
                                                                image_size,
                                                                embedding_size, nrof_images, nrof_batches, emb_array,
                                                                batch_size, image_paths)
                    # sorted_clusters = cluster_facial_encodings(facial_encodings)
                    sorted_clusters = cluster_encodings(facial_encodings, 6)
                    if not sorted_clusters:
                        cluster_centers = []
                        continue

                    # Remove the faces from the same image. Since they cannot came from same person,
                    # remove least-distance one from the top cluster
                    # sorted_clusters = data_cleaning(sorted_clusters, facial_encodings)

                    # print('Start zip upload for: ' + str(fam_id))
                    # futures.append(executor.submit(zip_and_upload, sorted_clusters, fam_id))
                    print('Saving result for sorted clusters!...' + str(fam_input) + '/' + job)

                    if children > 1:
                        # if not cluster_centers:
                        #     for cluster in sorted_clusters[:children]:
                        #         cluster_centers.append(compute_cluster_center(cluster, facial_encodings))
                        # else:
                        #     cluster_to_search = sorted_clusters[:children]
                        #     cluster_encodings = []
                        #     for cluster in cluster_to_search:
                        #         encodings = [facial_encodings[path] for path in cluster]
                        #         cluster_encodings.append(encodings)
                        #     reorder = []
                        #     for center in cluster_centers:
                        #         max_dis = 0
                        #         max_idx = 0
                        #         for idx, encoding in enumerate(cluster_encodings):
                        #             if idx in reorder:
                        #                 continue
                        #             dis = np.mean(face_distance(center, encoding))
                        #             if dis > max_dis:
                        #                 max_dis = dis
                        #                 max_idx = idx
                        #         reorder.append(max_idx)
                        #     sorted_clusters = [cluster_to_search[i] for i in reorder]
                        #     cluster_centers = []
                        for idx, cluster in enumerate(sorted_clusters):
                            # cluster_centers.append(compute_cluster_center(cluster, facial_encodings))
                            futures.append(executor.submit(save_one_cluster, os.path.join(fam_output, job, str(idx)), cluster))
                    else:
                        futures.append(executor.submit(save_one_cluster, os.path.join(fam_output, job, str(0)), sorted_clusters[0]))

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            print('job is finished!: ' + future.result())
                        except Exception as e:
                            print('zip and upload job failed!: ' + str(e))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Get a face input series (t-pose)')
    parser.add_argument('--input', type=str, help='input dir', required=True)
    parser.add_argument('--output', type=str, help='output dir', required=True)
    parser.add_argument('--children', type=int, help='children count for family', required=True)
    args = parser.parse_args()
    return args

args = parse_args()
run(args.input, args.output, args.children)