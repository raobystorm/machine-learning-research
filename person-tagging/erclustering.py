# Python 2.7

import os
import cPickle
import erclustering

base_folder = '/home/centos/mitene-pre_experiment/results'
ref_set = []
ref_filename_list = []
test_set = []
test_filename_list = []

F = 4096
gallery_size = 50
K = 200

for sub_folder in os.listdir(base_folder):
    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_reference.dat', 'rb') as f:
        ref_raw = cPickle.load(f)

    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_test.dat', 'rb') as f:
        test_raw = cPickle.load(f)

    print 'Reference set and test set loaded! Start clustering for ' + sub_folder + ' data set.'

    G = len(ref_raw)
    for array, img_name in ref_raw:
        ref_set.append(array)
        ref_filename_list.append(img_name)
    ref_set = map(list, zip(*ref_set))
    
    N = len(test_raw)
    for array, img_name in test_raw:
        test_set.append(array)
        test_filename_list.append(img_name)
    test_set = map(list, zip(*test_set))

    result = erclustering.Rank1Count(test_set, ref_set, G, N, F, K, gallery_size)
