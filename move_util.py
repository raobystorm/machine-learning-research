import os
import csv

with open('/Users/rui.zhong/audio-recognition/AudioSet/balanced_train_segments.csv', mode='r') as fp:
    with open('/Users/rui.zhong/audio-recognition/AudioSet/eval_segments.csv') as eval_fp:

        reader = csv.reader(fp)
        str_list = []
        for row in reader:
            str_list.append(row)
        eval_list = []
        reader = csv.reader(eval_fp)
        for row in reader:
            eval_list.append(eval_fp)

        for filename in os.listdir('/Users/rui.zhong/audio-recognition/AudioSet/processed/music'):
            filename = filename.rsplit('.', 1)[0]
            if list(filter(lambda x: filename in x, eval_list)):
                os.rename('/Users/rui.zhong/audio-recognition/AudioSet/processed/music/' + filename + '.wav', '/Users/rui.zhong/audio-recognition/AudioSet/eval_music/' + filename + '.wav')
                #os.rename('/Users/rui.zhong/audio-recognition/AudioSet/music/' + music + '.wav', '/Users/rui.zhong/audio-recognition/AudioSet/nonmusic/' + music+ '.wav')

        print('Finish')
        '''
        for nonmusic in os.listdir('/Users/rui.zhong/audio-recognition/AudioSet/nonmusic'):
            nonmusic = nonmusic.rsplit('.', 1)[0]
            if '/m/04rlf' in str(list(filter(lambda x: nonmusic in x, str_list))[0]):
                os.rename('/Users/rui.zhong/audio-recognition/AudioSet/nonmusic/' + nonmusic+ '.wav', '/Users/rui.zhong/audio-recognition/AudioSet/music/' + nonmusic+ '.wav')
        '''
