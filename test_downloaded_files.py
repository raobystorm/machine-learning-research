import csv
import os

with open('/home/centos/audio-recognition/AudioSet/unbalanced_train_segments.csv', mode='r') as fp:
    reader = csv.reader(fp)
    str_list = []
    for row in reader:
        str_list.append(row)

    for filename in os.listdir('/home/centos/audio-recognition/AudioSet/music'):
        filename = filename.rsplit('.', 1)[0]
        assert '/m/04rlf' in str(list(filter(lambda x: filename in x, str_list))[0])

    for filename in os.listdir('/home/centos/audio-recognition/AudioSet/nonmusic'):
        filename = filename.rsplit('.', 1)[0]
        assert '/m/04rlf' not in str(list(filter(lambda x: filename in x, str_list))[0])
