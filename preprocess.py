import os
import librosa
import pickle
import dill
from datetime import datetime
import time

import multiprocessing as mp

random_sample_size = 256
base_url = '/home/centos/audio-recognition/AudioSet'
music_files_limit = 22000
nonmusic_files_limit = 22000
music_files_path = base_url + '/music'
processed_music_files_path = base_url + '/processed/music'
nonmusic_files_path = base_url + '/nonmusic'
processed_nonmusic_files_path = base_url + '/processed/nonmusic'
#music_files_limit = 100
#nonmusic_files_limit = 100
#music_files_path = base_url + '/eval_music'
#processed_music_files_path = base_url + '/processed/eval_music'
#nonmusic_files_path = base_url + '/eval_nonmusic'
#processed_nonmusic_files_path = base_url + '/processed/eval_nonmusic'

job_list = []

def consume(in_q, out_q):
    while True:
        try:
            job = in_q.get()
            if job is None:
                break
            print('process %s' % job[0])
            y, sr = librosa.load(job[0], sr=44100)
            if len(y) is not 0:
                mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
                mfcc = mfcc.transpose()
                # For some samples the length is insufficient, just ignore them
                if len(mfcc) >= random_sample_size:
                    os.rename(job[1] + '/' + job[0], job[2] + '/' + job[0])
                    out_q.put([mfcc, job[3]])
        except:
            pass

"""
class Job(object):
    def __init__(self, filename, file_path, processed_files_path, is_music):
        self.filename = filename
        self.is_music = is_music
        self.file_path = file_path
        self.processed_files_path = processed_files_path
"""

def main():

    in_q = mp.Queue()
    out_q = mp.Queue()

    for filename in os.listdir(music_files_path):
        print('into input queue: %s' % music_files_path + '/' + filename)
        await in_q.put((filename, music_files_path, processed_music_files_path, [1., 0.]))

    for filename in os.listdir(nonmusic_files_path):
        print('into input queue: %s' % nonmusic_files_path + '/' + filename)
        await in_q.put((filename, nonmusic_files_path, processed_nonmusic_files_path, [0., 1.]))

    workers = [mp.Process(target=consume, args=(in_q, out_q,)) for i in range(4)]

    for i in workers:
        i.start()

    for i in wokers:
        in_q.put(None)

    wait(workers)
    out_q.put(None)

    persistance(out_q)


def persistance(q):
    limit = 4000
    dump_list = []
    stop = False
    print('process queue!')
    while True:
        with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
            count = 0
            if stop:
                break
            while count < limit:
                #with open(base_url + '/eval_data.dat', 'wb') as fp:
                    feature = q.get()
                    if feature is None:
                        stop = True
                        break
                    if len(feature[0]) >= 256:
                        dump_list.append(feature)
                        count += 1
            dill.dump(dump_list, fp)


if __name__ == '__main__':
    main()
