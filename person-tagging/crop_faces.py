# Python3.6.1

from PIL import Image
import os

base_folder = '/Users/rui.zhong/mitene-pre_experiment'
input_file = '/Users/rui.zhong/results.txt'

size_threshold = 50

img_sets = {}
for sub_folder in os.listdir(base_folder):
    sub_set = set()
    if sub_folder == '.DS_Store' or sub_folder == 'face_images':
        continue
    for img in os.listdir(base_folder + '/' + sub_folder + '/images'):
        if os.path.splitext(base_folder + '/' + sub_folder + '/images/' + img)[1] != '.jpg':
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
        x = int(round(float(items[1].split(':')[1])))
        y = int(round(float(items[2].split(':')[1])))
        width = int(round(float(items[3].split(':')[1])))
        height = int(round(float(items[4].split(':')[1])))
        score = float(items[5].split(':')[1])
        print
        if score < 0.9:
            continue
        if width <= size_threshold or height <= size_threshold:
            continue 
        sub_folder = find_set(img_sets, filename)
        if sub_folder == None:
            continue
        print('processing: ' + base_folder + '/' + sub_folder + '/images/' + filename)
        img = Image.open(base_folder + '/' + sub_folder + '/images/' + filename)
        img_crop = img.crop((x, y, x + width, y + height))
        img_crop = img_crop.resize((160, 160))
        save_path = base_folder + '/' + sub_folder + '/face_images/' + os.path.splitext(filename)[0] + '_' + str(x) + '_' + str(y) + '_' + str(width) + '_' + str(height) + '.jpg'
        img_crop.save(save_path)
        print('saved: ' + save_path)