# python3.6

import sys
import os
import numpy
import random
import matplotlib.pyplot as plt

base_folder = '/Users/rui.zhong/Downloads/home/mixi-mitene-cv/jupyter/mitene-pre_experiment/results'
sub_folder = os.listdir(base_folder)[2]

res_file = base_folder + '/' + sub_folder + '/result_' + sub_folder + '.dat'
with open(res_file, 'r') as f:
    strs = f.readlines()

img_filename = strs[0].replace('\n', '').split(',')
strs = strs[1:]
mat = []
for line in strs:
    nums = line.replace(' ', '').split(',')
    nums = [ float(f) for f in nums ]
    mat += nums

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(mat, bins=200, color='cyan', alpha=0.5)

ax.set_title(sub_folder)
fig.show()
input()