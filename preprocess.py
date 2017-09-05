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


manager = mp.Manager()
input_q = manager.Queue()
output_q = manager.Queue()

def process_one_file():
    while True:
        job_ = input_q.get()
        job = dill.loads(job_)
        if input_q.empty():
            break;
        print('process file in queue:' % job.filename)
        try:
            y, sr = librosa.load(job.filename, sr=44100)
            if len(y) is not 0:
                mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
                mfcc = mfcc.transpose()
                # For some samples the length is insufficient, just ignore them
                if len(mfcc) >= random_sample_size:
                    if job.is_music:
                        output_q.put([mfcc, [1., 0.]])
                    else:
                        output_q.put([mfcc, [0., 1.]])
                    os.rename(job.file_path + '/' + job.filename, job.processed_files_path + '/' + job.filename)
        except:
            pass
    if q.empty():
        output_q.put(int(-1))

class Job(object):
    def __init__(self, filename, file_path, processed_files_path, is_music):
        self.filename = filename
        self.is_music = is_music
        self.file_path = file_path
        self.processed_files_path = processed_files_path

def main():
    for filename in os.listdir(music_files_path):
        print('into input queue: %s' % music_files_path + '/' + filename)
        input_q.put(dill.dumps(Job(filename, music_files_path, processed_music_files_path, True)))

    for filename in os.listdir(nonmusic_files_path):
        print('into input queue: %s' % nonmusic_files_path + '/' + filename)
        input_q.put(dill.dumps(Job(filename, nonmusic_files_path, processed_nonmusic_files_path, False)))

    with mp.Pool(processes=4) as pool:
        pool.apply_async(process_one_file)

    persistance()

def persistance():
    limit = 4000
    dump_list = []
    stop = False
    print('process output queue!')
    while True:
        count = 0
        if stop:
            break
        while count < limit:
            with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
            #with open(base_url + '/eval_data.dat', 'wb') as fp:
                feature = output_q.get()
                if isinstance(feature, int):
                    stop = True
                    break
                if len(feature[0]) >= 256:
                    dump_list.append(feature)
                    count += 1
        dill.dump(dump_list, fp)


if __name__ == '__main__':
    main()
