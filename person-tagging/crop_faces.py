# Python3.6.1

from PIL import Image
import os

base_folder = '/Users/rui.zhong/mitene-pre_experiment'
input_file = '/Users/rui.zhong/results.txt'


img_sets = {}
for sub_folder in os.listdir(base_folder):
    sub_set = set()
    if sub_folder == '.DS_Store':
        continue
    for img in os.listdir(base_folder + '/' + sub_folder):
        if os.path.splitext(base_folder + '/' + sub_folder + '/' + img)[1] != '.jpg':
            continue
        sub_set.add(img)
    img_sets[str(sub_folder)] = sub_set

def find_set(img_sets, img):
    for key, sets in img_sets.items():
        if img in sets:
            return key

with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.replace(' ', '').split(',')
        filename = items[0]
        print('start processing: ' + filename)
        x = int(items[1].split(':')[1])
        y = int(items[2].split(':')[1])
        width = int(items[3].split(':')[1])
        height = int(items[4].split(':')[1])
        sub_folder = find_set(img_sets, filename)
        if sub_folder == None:
            continue
        print('processing: ' + base_folder + '/' + sub_folder + '/' + filename)
        img = Image.open(base_folder + '/' + sub_folder + '/' + filename)
        img_crop = img.crop((x, y, x + width, y + height))
        save_path = base_folder + '/' + sub_folder + '/results/' + os.path.splitext(filename)[0] + '_' + str(x) + '.jpg'
        img_crop.save(save_path)
        print('saved: ' + save_path)