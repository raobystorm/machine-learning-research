import csv
import os

with open('/home/centos/audio-recognition/AudioSet/balanced_train_segments.csv', mode='r') as fp:
    with open('/home/centos/audio-recognition/AudioSet/eval_segments.csv', mode='r') as eval_fp:
        reader = csv.reader(fp)
        str_list = []
        for row in reader:
            str_list.append(row)

        reader = csv.reader(eval_fp)
        eval_list = []
        for row in reader:
            eval_list.append(row)

        for filename in os.listdir('/home/centos/audio-recognition/AudioSet/music'):
            filename = filename.rsplit('.', 1)[0]
            assert '/m/04rlf' in str(list(filter(lambda x: filename in x, str_list))[0])

        for filename in os.listdir('/home/centos/audio-recognition/AudioSet/nonmusic'):
            filename = filename.rsplit('.', 1)[0]
            assert '/m/04rlf' not in str(list(filter(lambda x: filename in x, str_list))[0])

        print('train data OK')

        for filename in os.listdir('/home/centos/audio-recognition/AudioSet/eval_music'):
            filename = filename.rsplit('.', 1)[0]
            assert '/m/04rlf' in str(list(filter(lambda x: filename in x, eval_list))[0])

        for filename in os.listdir('/home/centos/audio-recognition/AudioSet/eval_nonmusic'):
            filename = filename.rsplit('.', 1)[0]
            assert '/m/04rlf' not in str(list(filter(lambda x: filename in x, eval_list))[0])

        print('eval data OK')
