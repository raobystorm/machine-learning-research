from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from PIL import Image
import zipfile
import boto3
import concurrent.futures
from shutil import rmtree


# There is a possibility that people have obscured face by each other. This
# will result same result face idx for different bodies. We decide to skip those
# bodies because the face to body bound is very week and ambiguous.
def find_face_crop(uuid, point, face_det_list):
    if uuid not in face_det_list:
        return None, None
    for i, ext, x1, y1, x2, y2 in face_det_list[uuid]:
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        width = x2 - x1
        height = y2 - y1
        if x1 + 0.2 * width < point[0] < x2 - 0.2 * width and y1 + 0.2 * height < point[1] < y2 - 0.2 * height:
            return i, ext
    return None, None


def get_bbox_from_area(keypoints):
    min_x = 9999999
    max_x = 0
    min_y = 9999999
    max_y = 0
    for i in range(len(keypoints) // 3):
        x, y, confidence = keypoints[i*3:(i+1)*3]
        if x == y == 0:
            continue
        if confidence < 0.4:
            continue

        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    return (min_x, min_y, max_x, max_y)


# face_det_list is a list contains all face detection results for this family
def process_one_family(fam_id, args, faces_det_list):
    pose_folder = os.path.join(args.pose_dir, fam_id)
    body_image_folder = os.path.join(args.body_dir, fam_id)
    image_folder = os.path.join(args.image_dir, fam_id)
    image_filenames = list(os.listdir(image_folder))
    save_paths = []

    if os.path.exists(body_image_folder):
        rmtree(body_image_folder)
    os.mkdir(body_image_folder)

    for pose_file in os.listdir(pose_folder):
        with open(os.path.join(pose_folder, pose_file), 'r') as f:
            json_body = json.loads(f.read())

        uuid = pose_file.split('_')[0]
        original_filename = list(filter(lambda x: x.startswith(uuid), image_filenames))[0]
        img = Image.open(os.path.join(image_folder, original_filename))
        for i, person in enumerate(json_body['people']):
            body_points = person["pose_keypoints_2d"]
            idx, ext = find_face_crop(uuid, (body_points[0], body_points[1]), faces_det_list)
            if idx is None:
                save_name = uuid + '_body_' + str(i) + '.jpeg'
            else:
                save_name = uuid + '_' + idx + '.' + ext

            min_x, min_y, max_x, max_y = get_bbox_from_area(body_points)
            f_min_x, f_min_y, f_max_x, f_max_y = get_bbox_from_area(person["face_keypoints_2d"])
            min_x = min(min_x, f_min_x)
            min_y = min(min_y, f_min_y)
            max_x = max(max_x, f_max_x)
            max_y = max(max_y, f_max_y)
            width = max_x - min_x
            height = max_y - min_y
            if width < 20 or height < 20:
                continue
            min_x = max(min_x - 0.2 * width, 0)
            min_y = max(min_y - 0.2 * height, 0)
            max_x = min(max_x + 0.2 * width, img.size[0])
            max_y = min(max_y + 0.2 * height, img.size[1])
            crop_body = img.crop((min_x, min_y, max_x, max_y))
            save_path = os.path.join(body_image_folder, save_name)

            if save_path in save_paths:
                save_paths.remove(save_path)
                continue

            crop_body.save(save_path)
            save_paths.append(save_path)

    filename = os.path.join('/tmp/', fam_id + '.zip')
    with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for save_path in save_paths:
            zf.write(filename=save_path, arcname=os.path.join(fam_id, os.path.basename(save_path)))

    s3 = boto3.session.Session().resource('s3')
    s3.Bucket('mitene-deeplearning-dataset').upload_file(filename, 'body/' + os.path.basename(filename))
    return fam_id


def process_one_family_pose(fam_id, pose_folder):
    filename = os.path.join('/tmp', fam_id + '_pose.zip')
    with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for json_file in os.listdir(pose_folder):
            zf.write(filename=os.path.join(pose_folder, json_file), arcname=os.path.join(fam_id, json_file))

    s3 = boto3.session.Session().resource('s3')
    s3.Bucket('mitene-deeplearning-dataset').upload_file(filename, 'pose/' + os.path.basename(filename))
    return fam_id


def run(args):
    face_bbox_file = '/home/ubuntu/face_list.txt'
    children_face_file = '/home/ubuntu/children_faces.txt'
    uuid2famid = {}
    faces = {}
    children_faces = set()

    # Setup uuid to fam_id dict
    for fam_id in os.listdir(args.image_dir):
        fam_folder = os.path.join(args.image_dir, fam_id)
        for img_path in os.listdir(fam_folder):
            uuid2famid[os.path.splitext(img_path)[0]] = fam_id

    print('Finished setting up uuid to fam_id dict!')

    with open(children_face_file, 'r') as f:
        for line in f.readlines():
            children_faces.add(line[:-1])

    # Read all face bbox data into mem
    with open(face_bbox_file, 'r') as f:
        for line in f.readlines():
            filename, items = line.split(':')
            filename, ext = filename.split('.')
            if filename not in children_faces:
                continue
            uuid, idx = filename.split('_')
            items = items.split(',')
            x1 = int(round(float(items[0])))
            y1 = int(round(float(items[1])))
            x2 = int(round(float(items[2])))
            y2 = int(round(float(items[3])))

            if uuid2famid[uuid] not in faces:
                faces[uuid2famid[uuid]] = {}

            if uuid not in faces[uuid2famid[uuid]]:
                faces[uuid2famid[uuid]][uuid] = []

            faces[uuid2famid[uuid]][uuid].append((idx, ext, x1, y1, x2, y2))

    print('Finished reading face bbox file!')

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for fam_id in faces.keys():
            futures.append(executor.submit(process_one_family, fam_id, args, faces[fam_id]))

        for future in concurrent.futures.as_completed(futures):
            try:
                print('job is finished for body! ' + future.result())
            except Exception as e:
                print('job failed for body image extraction! ' + str(e))

        # futures = []
        # for fam_id in os.listdir(args.pose_dir):
        #     futures.append(executor.submit(process_one_family_pose, fam_id, os.path.join(args.pose_dir, fam_id)))
        #
        # for future in concurrent.futures.as_completed(futures):
        #     try:
        #         print('job is finished for pose! ' + future.result())
        #     except Exception as e:
        #         print('job failed for pose zip and upload! ' + str(e))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Get a family photo input series (t-pose)')
    parser.add_argument('--image_dir', type=str, help='input original image dir', required=True)
    parser.add_argument('--pose_dir', type=str, help='input pose json dir', required=True)
    parser.add_argument('--body_dir', type=str, help='output body image crop dir', required=True)
    args = parser.parse_args()
    return args


run(parse_args())