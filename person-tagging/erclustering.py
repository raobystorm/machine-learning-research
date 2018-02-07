# Python 2.7

import os
import cPickle
import random
import erclustering

base_folder = '/home/centos/mitene-pre_experiment/results'
ref_set = []
ref_filename_list = []
test_set = []
test_filename_list = []

F = 4096
gallery_size = 100
K = 200

def nonzerorization(mat):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] == 0.0:
                mat[i][j] = random.random() * 1e-5

def get_data_from_file(f):
    raw = cPickle.load(f)
    count = len(raw)
    data_set = []
    filename_list = []
    for array, img_name in raw:
        data_set.append(array)
        filename_list.append(img_name)
    data_set = map(list, zip(*data_set))
    nonzerorization(data_set)
    return data_set, filename_list, count

for sub_folder in os.listdir(base_folder):
    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_reference.dat', 'rb') as f:
        ref_raw, ref_filename_list, G = get_data_from_file(f)

    with open(base_folder + '/' + sub_folder + '/data_' + sub_folder + '_test.dat', 'rb') as f:
        test_raw, test_filename_list, N = get_data_from_file(f)

    print 'Reference set and test set loaded! Start clustering for ' + sub_folder + ' data set.'
    result = erclustering.Rank1Count(test_raw, ref_raw, G, N, F, K, gallery_size)

    with open(base_folder + '/' + sub_folder + '/result_' + sub_folder + '.dat', 'w') as f:
        for filename in test_filename_list:
            f.write(filename)
            f.write(',')
        f.write('\n')
        for res in result:
            f.write(str(res).replace('[', '').replace(']', ''))
            f.write('\n')