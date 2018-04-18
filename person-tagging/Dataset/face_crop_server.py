
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import sys
import skimage
import skimage.io
from shutil import rmtree, copytree

# Module constant definition
ImageTypes = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.png')
crop_size = (160, 160)

# Import caffe and initialization
caffe_root = '/home/ubuntu/caffe-sfd/'
sys.path.insert(0, caffe_root + 'python')
import caffe

if not caffe:
    print('# Error: caffe-sfd import failure!')
    sys.exit(-1)


def _load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.
    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).
    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    original_img = skimage.io.imread(filename, as_grey=not color)
    # Because we need original image for saving cropped one, force copy
    img = skimage.img_as_float(original_img, force_copy=True).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img, original_img


def _detect_face(net, image, shrink):
    if shrink != 1:
        image = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']
    det_conf = detections[0, 0, :, 2]
    det_xmin = image.shape[1] * detections[0, 0, :, 3] / shrink
    det_ymin = image.shape[0] * detections[0, 0, :, 4] / shrink
    det_xmax = image.shape[1] * detections[0, 0, :, 5] / shrink
    det_ymax = image.shape[0] * detections[0, 0, :, 6] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def _multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = _detect_face(net, image, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = _detect_face(net, image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, _detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, _detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def _flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = _detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def _bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


def _process(net, job, det_threshold=0.9, size_threshold=50):
    img_folder = '/home/ubuntu/images/' + job
    mvdir_des = '/home/ubuntu/images_backup/' + job
    output = '/home/ubuntu/faces/' + job

    if not os.path.exists(output):
        os.makedirs(output)

    for img_name in os.listdir(img_folder):
        if os.path.splitext(img_name)[1] not in ImageTypes:
            continue
        # Use caffe-sfd to detect faces in images
        caffe_image, original_img = _load_image(img_folder + '/' + img_name)
        max_im_shrink = (0x7fffffff / 577.0 / (caffe_image.shape[0] * caffe_image.shape[1])) ** 0.5  # the max size of input image for caffe
        shrink = max_im_shrink if max_im_shrink < 1 else 1
        det0 = _detect_face(net, caffe_image, shrink)
        det1 = _flip_test(net, caffe_image, shrink)
        [det2, det3] = _multi_scale_test(net, caffe_image, max_im_shrink)
        det = np.row_stack((det0, det1, det2, det3))
        dets = _bbox_vote(det)
        dets = dets[np.where(dets[:, 4] >= det_threshold)]
        for i in range(dets.shape[0]):
            xmin = max(int(dets[i][0]), 0)
            ymin = max(int(dets[i][1]), 0)
            xmax = int(dets[i][2])
            ymax = int(dets[i][3])
            # score = dets[i][4]
            # Ignore those does not have enough size
            if xmax - xmin < size_threshold or ymax - ymin < size_threshold:
                continue
            img = original_img[ymin:ymax, xmin: xmax]
            img = skimage.transform.resize(img, crop_size)
            name, ext = os.path.splitext(img_name)
            skimage.io.imsave(output + '/' + name + '_' + str(i) + ext, img)

        print('Detected {0} faces in :{1}'.format(dets.shape[0], img_name))

    print('Finished detection faces in family {0}!'.format(job))

    copytree(img_folder, mvdir_des)
    rmtree(img_folder, ignore_errors=True)


def run():

    caffe.set_mode_gpu()
    network = caffe_root + 'models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt'
    model = caffe_root + 'models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel'
    net = caffe.Net(network, model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    net.blobs['data'].reshape(1, 3, 640, 640)
    print('Network initialization finished! Start process jobs!')

    for dir in os.listdir('/home/ubuntu/images'):
        _process(net, dir)

    print('Finished all faces! Shutdown server...')


run()