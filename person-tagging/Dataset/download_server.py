
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Pool
import json
from logging import getLogger


def write_to_file(bytes_to_write, path):
    with open(path, 'wb') as f:
        f.write(bytes_to_write)


def handler(logger, request, exception):
    logger.error("Request exception occured!")
    logger.error("Request:" + str(request))
    logger.error("Exception:" + str(exception))


def get_instance_name():
    import requests
    response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
    ins_id = response.text
    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(id=ins_id)
    return [x['Value'] for x in instance.tags if x['Key'] == 'Name'][0]


def parse_instance_name(name):
    import boto3
    tags = name.split('-')
    file_number = tags[-1]
    image_size = tags[-2]
    sess = boto3.Session(region_name='ap-northeast-1')
    s3 = sess.resource('s3')
    bucket = 'mitene-deeplearning-dataset'
    file_key = 'media_list_files/' + file_number + '.json'
    download_file = '/home/ubuntu/download_image_list.json'
    s3.Object(bucket, file_key).download_file(download_file)
    return download_file, image_size


def run(logger=getLogger(__name__)):
    # Get the latest instance-id and name tag for instance
    name = get_instance_name()
    logger.info('Instances identified as :' + name)
    if not name:
        exit(-1)

    input_file, image_size = parse_instance_name(name)
    output = '/home/ubuntu/images'
    logger.info('Media file list has been downloaded as:' + input_file)

    from os.path import exists
    from os import makedirs
    from shutil import rmtree

    if exists(output):
        rmtree(output, ignore_errors=True)
    makedirs(output)

    image_size = '/' + image_size + '/'
    # This is the very secret signature that should not be saved any other place
    sig = '?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9tZWRpYS1maWxlcy1wcm9kdWN0aW9uLWNkbi5taXRlbmUudXMvbWVkaWEvdXBsb2Fkcy8qIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNTI0MDIyMzYxfX19XX0_&Signature=LacoJ8x1SSeLIL3T-BDx73pf27mgzpAcd5GgfeWhWksiY9t77XxzSXhQvxoJPH3RPY8S25hNbBVs2iI5fu4pqj8x-goADTJ20NBJBHubv~PRDeN3kYXbKKU89Jl-~a6p1YTD2p1i8Zs-dqXoiRgrKqe02zO3~lltq18bygEF-0CTC8Js62HohJ~ovDhW3UymuhjgHeTIIMFPlR6NFqpOSfwH0a6-ZksDjS4wHcicDPvZU9Uo208stuThtyefAGARjwHyEhyL5KQOqQWAky2Te-LuyFxX-bkOlfpU8nK62jt9LiRHdPDjxoz6SfATT-qnuK2g7M8Uvz-DHCR77PsEOw__&Key-Pair-Id=APKAJL6OCBMEEECHRPSQ'
    with open(input_file, 'r') as f:
        # DO NOT import grequests before download of s3 object. Library conflicts
        import grequests
        json_file = json.loads(f.read())
        [makedirs(output + '/' + key) for key in json_file.keys()]
        import time
        time.sleep(1)
        for i, key in enumerate(json_file.keys()):
            jobs = {}
            if not json_file[key]:
                continue
            for uuid in json_file[key]:
                jobs['https://media-files-production-cdn.mitene.us/media/uploads' + image_size + uuid + sig] = output + '/' + key + '/' + uuid
            responses = grequests.map((grequests.get(u) for u in jobs.keys()), size=120, exception_handler=handler)
            responses = list(filter(lambda x: x and x.status_code == 200, responses))
            with Pool(processes=8) as pool:
                res = [pool.apply_async(write_to_file, (res.content, jobs[res.url]),) for res in responses]
                logger.info("Finished download {0} files!".format(len([r.get() for r in res])))
            logger.info("Finidhed {0} / {1} families!".format(i + 1, len(json_file.keys())))
