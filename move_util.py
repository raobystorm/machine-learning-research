import os
import csv

base_url = '/home/centos/audio-recognition/AudioSet/'

with open(base_url + 'balanced_train_segments.csv', mode='r') as fp:
    with open(base_url + 'eval_segments.csv', mode='r') as eval_fp:

        #reader = csv.reader(fp)
        #str_list = []
        #for row in reader:
        #    str_list.append(row)
        eval_list = []
        reader = csv.reader(eval_fp)
        for row in reader:
            eval_list.append(row)

        for filename in os.listdir(base_url + 'eval_nonmusic'):
            filename = filename.rsplit('.', 1)[0]
            if '/m/04rlf' in str(list(filter(lambda x: filename in x, eval_list))[0]):
                os.rename(base_url + 'eval_nonmusic/' + filename + '.wav', base_url + '/' + filename + '.wav')
                #os.rename('/Users/rui.zhong/audio-recognition/AudioSet/music/' + music + '.wav', '/Users/rui.zhong/audio-recognition/AudioSet/nonmusic/' + music+ '.wav')

        print('Finish')
        '''
        for nonmusic in os.listdir(base_url + 'nonmusic'):
            nonmusic = nonmusic.rsplit('.', 1)[0]
            if '/m/04rlf' in str(list(filter(lambda x: nonmusic in x, str_list))[0]):
                os.rename(base_url + 'nonmusic/' + nonmusic+ '.wav', base_url + 'music/' + nonmusic+ '.wav')
        '''
