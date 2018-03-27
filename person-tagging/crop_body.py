from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from PIL import Image
from PIL import ImageDraw
import os
import pdb
import math

base_folder = '/Users/rui.zhong/result_body/user1'
img_folder = base_folder + '/images'
json_folder = base_folder + '/pose'
result_folder = base_folder + '/draw_bbox'
body_size = (128, 160)
arm_factor = 0.3

def draw_rect(img, vertices):
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6], vertices[7]
    draw = ImageDraw.Draw(img)
    point_list = []
    for x, y in ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)):
        if x != y != 0:
            point_list.append((x, y))

    draw.line(tuple(point_list), fill='rgb(255,0,0)')


def calc_torso_rectangle(vertices):
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6], vertices[7]
    if x2 == y2 == 0 or x3 == y3 == 0:
        return vertices
    k = (y3 - y2) / (x3 - x2)
    b = (x2 * y3 - x3 * y2) / x3 - x2


def calc_arm_rectangle(x1, y1, x2, y2):
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

    xx3 = x2 + dx
    xx4 = x2 - dx
    yy3 = k * xx3 + b2
    yy4 = k * xx4 + b2
    return (xx1, yy1, xx2, yy2, xx3, yy3, xx4, yy4)


for json_name in os.listdir(json_folder):
    if os.path.splitext(json_name)[-1] != '.json':
        continue

    with open(json_folder + '/' + json_name, 'r') as f:
        body = f.read()
        json_body = json.loads(body)

    img_name = json_name.split('_')[0]
    img = Image.open(img_folder + '/' + img_name + '.jpg')
    count = 0
    for person in json_body["people"]:
        # Draw torso
        points = person["pose_keypoints_2d"]
        x1, y1 = points[6], points[7]
        x2, y2 = points[24], points[25]
        x3, y3 = points[33], points[34]
        x4, y4 = points[15], points[16]
        if (x1 == y1 == 0) or (x4 == y4 == 0):
            continue
        if x2 == y2 == 0:
            x2 = x1
            y2 = min(img.size[0], y1 + abs(x4 - x1))
        if x3 == y3 == 0:
            x3 = x4
            y3 = min(img.size[0], y4 + abs(x4 - x1))

        draw_rect(img, [x1, y1, x2, y2, x3, y3, x4, y4])

        # Draw left arms
        l_x1, l_y1 = points[9], points[10]
        if l_x1 != l_y1 != 0:
            l_arm1_rect = calc_arm_rectangle(x1, y1, l_x1, l_y1)
            draw_rect(img, l_arm1_rect)
        l_x2, l_y2 = points[12], points[13]
        if l_x2 != l_y2 != 0:
            l_arm2_rect = calc_arm_rectangle(l_x1, l_y1, l_x2, l_y2)
            draw_rect(img, l_arm2_rect)
        
        # Draw right arms
        r_x1, r_y1 = points[18], points[19]
        if r_x1 != r_y1 != 0:
            r_arm1_rect = calc_arm_rectangle(x4, y4, r_x1, r_y1)
            draw_rect(img, r_arm1_rect)
        r_x2, r_y2 = points[21], points[22]
        if r_x2 != r_y2 != 0:
            r_arm2_rect = calc_arm_rectangle(r_x1, r_y1, r_x2, r_y2)
            draw_rect(img, r_arm2_rect)

    img.save(result_folder + '/' + img_name + '.jpg')