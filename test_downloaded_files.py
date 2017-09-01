import csv
import os

music_set = set()
nonmusic_set = set()

music_path = '/home/centos/audio-recognition/AudioSet/music'
nonmusic_path = '/home/centos/audio-recognition/AudioSet/nonmusic'

def read_csv_file(fp):
    reader = csv.reader(fp)
    str_list = []
    for row in reader:
        filename = row[0]
        string = ''.join(row)
        if '/m/04rlf' in string:
            music_set.add(filename + '.m4a')
        else:
            nonmusic_set.add(filename + '.m4a')

with open('/home/centos/audio-recognition/AudioSet/balanced_train_segments.csv', mode='r') as fp:
    read_csv_file(fp)

with open('/home/centos/audio-recognition/AudioSet/eval_segments.csv', mode='r') as fp:
    read_csv_file(fp)

with open('/home/centos/audio-recognition/AudioSet/unbalanced_train_segments.csv', mode='r') as fp:
    read_csv_file(fp)

for f in os.listdir(music_path):
    if f not in music_set and f in nonmusic_set:
        print('Move nonmusic in music folder! %s' % f)
        os.rename(music_path + '/' + f, nonmusic_path + '/' + f)

for f in os.listdir(nonmusic_set):
    if f not in nonmusic_set and f in music_set:
        print('Move music in nonmusic folder! %s' % f)
        os.rename(nonmusic_path + '/' + f, music_path + '/' + f)
