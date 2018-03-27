
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from PIL import Image
import math
import pdb
import numpy as np

size_threshold = 50
face_size = (160, 160)

body_size = (173, 227)
upper_arm_size = (27, 114)
lower_arm_size = (27, 113)
arm_factor = 0.3

original_base_folder = '/Users/rui.zhong/result_body/user1'
original_img_folder = original_base_folder + '/images'
pose_folder = original_base_folder + '/pose'
result_face_folder = original_base_folder + '/results/face'
result_body_folder = original_base_folder + '/results/body'
face_bbox_file = '/Users/rui.zhong/result_body/results.txt'

faces = {}

def find_face_crop(filename, point):
    if filename not in faces:
        return None
    for x, y, width, height in faces[filename]:
        if x < point[0] < x + width and y < point[1] < y + height:
            return x, y, x + width, y + height

def calc_face_from_shoulder(x1, y1, x2, y2):
    if x1 > x2:
        x1, x2 = x2, x1

    L = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dx = L / 2
    dy = L / 2
    xx = (x2 + x1) / 2
    yy = (y1 + y2) / 2 - L / 2

    return (xx - dx, yy - dy, xx + dx, yy + dy)


def get_bbox_from_area(keypoints):
    min_x = 9999999
    max_x = 0
    min_y = 9999999
    max_y = 0
    for i in range(len(keypoints) // 3):
        x, y, _ = keypoints[i * 3 : (i + 1) * 3]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    return (min_x, min_y, max_x, max_y)


def calc_arm_rectangle(x1, y1, x2, y2):
    if x1 == y1 == 0 or x2 == y2 == 0:
        return None

    if y2 == y1:
        L = arm_factor * abs(x2 - x1)
        dy = L / 2
        return (x1, y1 - dy, x1, y1 + dy, x2, y2 + dy, x2, y2 - dy)

    k = (x1 - x2) / (y2 - y1)
    L = arm_factor * math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    theta = math.atan(k)

    dx = math.cos(theta) * L / 2
    dy = math.sin(theta) * L / 2

    b1 = y1 - k * x1
    b2 = y2 - k * x2

    xx1 = x1 - dx
    xx2 = x1 + dx
    yy1 = k * xx1 + b1
    yy2 = k * xx2 + b1

    xx3 = x2 - dx
    xx4 = x2 + dx
    yy3 = k * xx3 + b2
    yy4 = k * xx4 + b2
    return (xx1, yy1, xx3, yy3, xx4, yy4, xx2, yy2)


with open(face_bbox_file, 'r') as f:
    for line in f.readlines():
        items = line.replace(' ', '').split(',')
        filename = items[0]
        x = int(round(float(items[1].split(':')[1])))
        y = int(round(float(items[2].split(':')[1])))
        width = int(round(float(items[3].split(':')[1])))
        height = int(round(float(items[4].split(':')[1])))
        score = float(items[5].split(':')[1])
        if score < 0.9:
            continue

        if filename not in faces:
            faces[filename] = []

        faces[filename].append((x, y, width, height))


for filename in os.listdir(original_img_folder):

    keypoint_file = pose_folder + '/' + os.path.splitext(filename)[0] + '_keypoints.json'
    img = Image.open(original_img_folder + '/' + filename)

    with open(keypoint_file, 'r') as f:
        body = f.read()
        json_body = json.loads(body)

    count = 0
    for person in json_body["people"]:
        imarray = np.random.rand(body_size[0] + upper_arm_size[0] + lower_arm_size[0], body_size[1], 3) * 255
        body_img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        body_points = person["pose_keypoints_2d"]
        face_points = find_face_crop(filename, (body_points[0], body_points[1]))

        x1, y1 = body_points[6], body_points[7]
        x2, y2 = body_points[24], body_points[25]
        x3, y3 = body_points[33], body_points[34]
        x4, y4 = body_points[15], body_points[16]

        if x2 == y2 == 0:
            x2 = x1
            y2 = min(img.size[0], y1 + abs(x4 - x1))
        if x3 == y3 == 0:
            x3 = x4
            y3 = min(img.size[0], y4 + abs(x4 - x1))

        trans_body = img.transform(body_size, Image.QUAD, (x1, y1, x2, y2, x3, y3, x4, y4))
        body_img.paste(trans_body, (upper_arm_size[0], 0))

        if (x1 == y1 == 0) or (x4 == y4 == 0):
            continue

        if face_points:
            crop_face = img.crop(face_points)
        else:
            face_candidate = person["face_keypoints_2d"]
            if all(v == 0 for v in face_candidate):
                crop_face = img.crop(calc_face_from_shoulder(x1, y1, x4, y4))
            else:
                crop_face = img.crop(get_bbox_from_area(face_candidate))

        crop_face = crop_face.resize(face_size)
        crop_face.save(result_face_folder + '/' + filename + '_' + str(count) + '_face.jpg')

        l_x1, l_y1 = body_points[9], body_points[10]
        l_arm1_rect = calc_arm_rectangle(x1, y1, l_x1, l_y1)
        if l_arm1_rect:
            trans_upper_left_arm = img.transform(upper_arm_size, Image.QUAD, l_arm1_rect)
            body_img.paste(trans_upper_left_arm, (0, 0))

        l_x2, l_y2 = body_points[12], body_points[13]
        l_arm2_rect = calc_arm_rectangle(l_x1, l_y1, l_x2, l_y2)
        if l_arm2_rect:
            trans_lower_left_arm = img.transform(lower_arm_size, Image.QUAD, l_arm2_rect)
            body_img.paste(trans_lower_left_arm, (0, upper_arm_size[1]))

        r_x1, r_y1 = body_points[18], body_points[19]
        r_arm1_rect = calc_arm_rectangle(x4, y4, r_x1, r_y1)
        if r_arm1_rect:
            trans_upper_right_arm = img.transform(upper_arm_size, Image.QUAD, r_arm1_rect)
            body_img.paste(trans_upper_right_arm, (body_size[0] + upper_arm_size[0], 0))

        r_x2, r_y2 = body_points[21], body_points[22]
        r_arm2_rect = calc_arm_rectangle(r_x1, r_y1, r_x2, r_y2)
        if r_arm2_rect:
            trans_lower_right_arm = img.transform(upper_arm_size, Image.QUAD, r_arm2_rect)
            body_img.paste(trans_lower_right_arm, (body_size[0] + upper_arm_size[0], upper_arm_size[1]))

        body_img.save(result_body_folder + '/' + filename + '_' + str(count) + '_body.jpg')
        count += 1