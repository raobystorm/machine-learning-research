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

def f(job):
    try:
        import librosa
        y, sr = librosa.load(job.filename, sr=44100)
        if len(y) is not 0:
            mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
            mfcc = mfcc.transpose()
            # For some samples the length is insufficient, just ignore them
            if len(mfcc) >= random_sample_size:
                if job.is_music:
                    f.q.put([mfcc, [1., 0.]])
                else:
                    f.q.put([mfcc, [0., 1.]])
                os.rename(job.file_path + '/' + job.filename, job.processed_files_path + '/' + job.filename)
    except:
        pass

class Job(object):
    def __init__(self, filename, file_path, processed_files_path, is_music):
        self.filename = filename
        self.is_music = is_music
        self.file_path = file_path
        self.processed_files_path = processed_files_path

def f_init(q):
    f.q = q

def main():

    cpus = mp.cpu_count()
    q = mp.Manager().Queue()

    for filename in os.listdir(music_files_path):
        print('into input queue: %s' % music_files_path + '/' + filename)
        job_list.append(Job(filename, music_files_path, processed_music_files_path, True))

    for filename in os.listdir(nonmusic_files_path):
        print('into input queue: %s' % nonmusic_files_path + '/' + filename)
        job_list.append(Job(filename, nonmusic_files_path, processed_nonmusic_files_path, False))

    q.put(job_list[0])
    pool = mp.Pool(cpus, f_init, [q])
    pool.imap(f, job_list)

    q.put(int(-1))
    persistance(q)

    pool.close()


def persistance(q):
    limit = 4000
    dump_list = []
    stop = False
    print('process queue!')
    with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
        while True:
            count = 0
            if stop:
                break
            while count < limit:
                #with open(base_url + '/eval_data.dat', 'wb') as fp:
                    feature = q.get()
                    if isinstance(feature, int):
                        stop = True
                        break
                    if len(feature[0]) >= 256:
                        dump_list.append(feature)
                        count += 1
            dill.dump(dump_list, fp)


if __name__ == '__main__':
    main()
