# Python 3.6

import sys
import os
import numpy
import random

labled_folder = '/Users/rui.zhong/nao_images'
res_file = labled_folder + '/result_nao.dat'
strs = []
threathhold = 110.0

# Convert Rank1Count matrix into adjacency matrix 'matrix'

def check_connectivity(str):
    f = float(str)
    return f >= threathhold

with open(res_file, 'r') as f:
    strs = f.readlines()
    filename_list = strs[0].replace(' ', '').replace('\n', '').split(',')[:-1]
    strs = strs[1:]

matrix = {}
for i in range(len(strs)):
    nums = strs[i].replace(' ', '').split(',')
    nums = [ check_connectivity(f) for f in nums ]
    mat = {}
    for j in range(len(filename_list)):
        mat[filename_list[j]] = nums[j]
    matrix[filename_list[i]] = mat

for i in filename_list:
    for j in filename_list:
        if matrix[i][j]:
            matrix[j][i] = True

# Use matrix T to grouping filenames

groups = []

def find(filename):
    for group in groups:
        if filename in group:
            return group

for i in filename_list:
    groups.append(set([i]))

for i in filename_list:
    for j in filename_list:
        if i == j:
            continue
        if matrix[i][j]:
            set_i = find(i)
            set_j = find(j)
            if set_i != set_j:
                groups.remove(set_i)
                groups.remove(set_j)
                groups.append(set_i.union(set_j))

with open(labled_folder + '/analysis_res.txt', 'w') as f:
    for group in groups:
        f.write(str(group))
        f.write('\n\n')

# Evaluation

tp = 0
fn = 0
fp = 0
tn = 0

def in_the_same_set(i, j):
    for any_set in sets:
        if i in any_set and j in any_set:
            return True
    return False

result_folder = '/Users/rui.zhong/nao_images/results'
for folder in os.listdir(result_folder):
    if folder == 'others':
        others_set = set(os.listdir(result_folder + '/' + folder))
    elif folder == 'child':
        child_set = set(os.listdir(result_folder + '/' + folder))
    elif folder == 'father':
        father_set = set(os.listdir(result_folder + '/' + folder))
    else:
        mother_set = set(os.listdir(result_folder + '/' + folder))

sets = [child_set, father_set, mother_set, others_set]

for i in filename_list:
    for j in filename_list:
        g_i = find(i)
        g_j = find(j)
        if i not in others_set and j not in others_set:
            if g_i == g_j and in_the_same_set(i, j):
                tp += 1
            elif in_the_same_set(i, j) and g_i != g_j:
                fn += 1
            elif (not in_the_same_set(i, j)) and g_i == g_j:
                fp += 1
            elif (not in_the_same_set(i, j)) and g_i != g_j:
                tn += 1
        elif i in others_set and j in others_set:
            if g_i != g_j:
                tp += 1
            elif g_i == g_j:
                fp += 1
        elif (i not in others_set) or (j not in others_set):
            if g_i == g_j:
                fp += 1
            elif g_i != g_j:
                tn += 1
        else:
            print('SHOULD NOT BE HERE!!!!!')

P = tp / (tp + fp)
R = tp / (tp + fn)
b = 2
F = (b**2 + 1) * P * R / (b**2 * P + R)
RI = (tp + tn) / (tp + fp + fn + tn)

print(P)
print(R)
print(F)
print(RI)