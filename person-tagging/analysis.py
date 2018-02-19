# Python 3.6

import sys
import os
import numpy
import random
import shutil

sub_folder = 'kasahara'
labled_folder = '/Users/rui.zhong/results/' + sub_folder + '_images/'
source_img_folder = '/Users/rui.zhong/mitene-pre_experiment_with_face/' + sub_folder + '/body_images'
res_file = labled_folder + '/result_' + sub_folder + '.dat'
result_folder = labled_folder + '/results'
strs = []
start_threas = 95.0
end_threas = 125.0
delta = 0.1
res = str('\n')

def check_connectivity(str, threashold):
    f = float(str)
    return f >= threashold


def no_do_not_link_constraint(set_i, set_j):
    for i in set_i:
        for j in set_j:
            if i.split('_')[0] == j.split('_')[0]:
                return False
    return True


def find(filename, groups):
    for group in groups:
        if filename in group:
            return group


def in_the_same_set(i, j, sets):
    for any_set in sets:
        if i in any_set and j in any_set:
            return True
    return False


def run_with_threas(threashold, res):
    # Convert Rank1Count matrix into adjacency matrix 'matrix'

    with open(res_file, 'r') as f:
        strs = f.readlines()
        filename_list = strs[0].replace(' ', '').replace('\n', '').split(',')[:-1]
        strs = strs[1:]

    matrix = {}
    for i in range(len(strs)):
        nums = strs[i].replace(' ', '').split(',')
        nums = [ check_connectivity(f, threashold) for f in nums ]
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

    for i in filename_list:
        groups.append(set([i]))

    for i in filename_list:
        for j in filename_list:
            if i == j:
                continue
            if matrix[i][j]:
                set_i = find(i, groups)
                set_j = find(j, groups)
                if set_i != set_j and no_do_not_link_constraint(set_i, set_j):
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

    for folder in os.listdir(result_folder):
        if folder == 'others':
            others_set = set(os.listdir(result_folder + '/' + folder))
        elif folder == 'child1':
            child_set = set(os.listdir(result_folder + '/' + folder))
        elif folder == 'child2':
            child2_set = set(os.listdir(result_folder + '/' + folder))
        elif folder == 'father':
            father_set = set(os.listdir(result_folder + '/' + folder))
        elif folder == 'mother':
            mother_set = set(os.listdir(result_folder + '/' + folder))
        else:
            continue

    sets = [child_set, child2_set, father_set, mother_set, others_set]

    for i in filename_list:
        for j in filename_list:
            if i == j:
                continue
            body_i = i.split('_')[0] + '_' + i.split('_')[1] + '.jpg'
            body_j = j.split('_')[0] + '_' + j.split('_')[1] + '.jpg'
            g_i = find(body_i, groups)
            g_j = find(body_j, groups)
            if body_i not in others_set and body_j not in others_set:
                if g_i == g_j and in_the_same_set(body_i, body_j, sets):
                    tp += 1
                elif in_the_same_set(body_i, body_j, sets) and g_i != g_j:
                    fn += 1
                elif (not in_the_same_set(body_i, body_j, sets)) and g_i == g_j:
                    fp += 1
                elif (not in_the_same_set(body_i, body_j, sets)) and g_i != g_j:
                    tn += 1
            elif body_i in others_set and body_j in others_set:
                if g_i != g_j:
                    tp += 1
                elif g_i == g_j:
                    fp += 1
            elif (body_i not in others_set) or (body_j not in others_set):
                if g_i == g_j:
                    fp += 1
                elif g_i != g_j:
                    tn += 1
            else:
                print('SHOULD NOT BE HERE!!!!!')

    P = tp * 1.0 / (tp + fp)
    R = tp * 1.0 / (tp + fn)
    b = 2
    F = (b**2 + 1) * P * R / (b**2 * P + R)
    RI = (tp + tn) * 1.0 / (tp + fp + fn + tn)

    print('threathhold: ' + str(threashold))
    res += ('threathhold: ' + str(threashold) + '\n')
    print('precision: ' + str(P))
    res += ('precision: ' + str(P) + '\n')
    print('recall: ' + str(R))
    res += ('recall: ' + str(R) + '\n')
    print('F: ' + str(F))
    res += ('F: ' + str(F) + '\n')
    print('RI: ' + str(RI))
    res += ('RI: ' + str(RI) + '\n')

    largest = max(groups, key=len)
    correct = 0
    for i in largest:
        ii = i.split('_')[0] + '_' + i.split('_')[1] + '.jpg'
        if ii in child_set or ii in child2_set:
            correct += 1

    print('correct: ' + str(correct))
    res += ('correct: ' + str(correct) + '\n')
    print('accuracy: ' + str(correct * 1.0 / len(largest)) + '\n')
    res += ('accuracy: ' + str(correct * 1.0 / len(largest)) + '\n\n')
    return res, groups

# threashold = start_threas
# while threashold <= end_threas:
#     res, groups = run_with_threas(threashold, res)
#     threashold += delta

_, groups = run_with_threas(float(sys.argv[1]), res)

groups = sorted(groups, key=len)

with open(labled_folder + '/analysis_output.txt', 'w') as f:
    f.write(res)

grouped_fodler = labled_folder + '/grouped'

if os.path.exists(grouped_fodler):
    shutil.rmtree(grouped_fodler)

os.mkdir(grouped_fodler)

g_count = 0
for group in groups:
    group_path = grouped_fodler + '/group_' + str(g_count)
    os.mkdir(grouped_fodler + '/group_' + str(g_count))
    for img in group:
        img = img.split('_')[0] + '_' + img.split('_')[1] + '.jpg'
        shutil.copyfile(source_img_folder + '/' + img, group_path + '/' + img)
    g_count += 1