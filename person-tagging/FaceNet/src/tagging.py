from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil


def run(args):
    for fam_id in os.listdir(args.input):
        fam_folder = os.path.join(args.input, fam_id)
        tagging = {}
        original_image_folder = os.path.join(fam_folder, 'images')
        clustered_faces_folder = os.path.join(fam_folder, 'clustered')

        output_folder = os.path.join(fam_folder, 'result')
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)

        for job in os.listdir(clustered_faces_folder):
            job_folder = os.path.join(clustered_faces_folder, job)
            for cluster in os.listdir(job_folder):
                cluster_folder = os.path.join(job_folder, cluster)
                if cluster not in tagging:
                    tagging[cluster] = []
                tagging[cluster] += [os.path.basename(image).split('_')[0] for image in os.listdir(cluster_folder)]
                if not os.path.exists(os.path.join(output_folder, cluster)):
                    os.mkdir(os.path.join(output_folder, cluster))

        os.mkdir(os.path.join(output_folder, 'None'))

        for img in os.listdir(original_image_folder):
            uuid = os.path.splitext(img)[0]
            not_clustered_flag = True
            for idx, cluster in tagging.items():
                if uuid in cluster:
                    not_clustered_flag = False
                    shutil.copy2(os.path.join(original_image_folder, img), os.path.join(output_folder, idx))
            if not_clustered_flag:
                shutil.copy2(os.path.join(original_image_folder, img), os.path.join(output_folder, 'None'))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Get a face input series (t-pose)')
    parser.add_argument('--input', type=str, help='input dir', required=True)
    args = parser.parse_args()
    return args


run(parse_args())
