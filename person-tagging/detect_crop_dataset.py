from PIL import Image
import os
import json
import numpy as np
from object_detector import ObjectDetector
from server import DetectionJob
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input


base_folder = '/home/centos/mitene-pre_experiment'


def crop_person_clustering_dataset():
    input_json = base_folder + '/input_json.json'
    with open(input_json, 'r') as f:
        json_body = json.loads(f.read())
        for item in json_body:
            filename = item['filename']
            results = item['result']
            with Image.open(base_folder + '/' + filename) as img:
                if len(results[0]) == 0:
                    continue
                count = 0
                for result in results[0]:
                    x = result['x']
                    y = result['y']
                    width = result['width']
                    height = result['height']
                    img_crop = img.crop((x, y, x + width, y + height))
                    img_crop.save(base_folder + '/results/' + os.path.splitext(filename)[0] + '_' + str(count) + '.jpg')
                    count += 1


def detect_and_output():
    result_file = base_folder + '/input_json.json'
    model_path = 'VOC2012model_v1.hdf5'
    recognizer = ObjectDetector(model_path)
    results = []
    for sub_folder in os.listdir(base_folder):
        for filename in os.listdir(base_folder + '/' + sub_folder):
            extension = os.path.splitext(filename)[1]
            if extension != '.jpg':
                continue
            img = Image.open(base_folder + '/' + sub_folder + '/' + filename)
            sizes = [img.size]
            array = [img_to_array(img.resize((300, 300)))]
            array = preprocess_input(np.array(array))
            result = recognizer.recognize(DetectionJob(array=array, sizes=sizes, paths=[base_folder]))
            results.append({"filename": sub_folder + '/' + filename, "result": result})

    with open(result_file, 'w') as f:
        f.write(json.dumps(results))


if __name__ == '__main__':
    detect_and_output()
    crop_person_clustering_dataset()