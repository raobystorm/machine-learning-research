import os
import librosa
import numpy as np
import dill
from datetime import datetime
from audioread import NoBackendError
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
            y, sr = librosa.load(job[1] + '/' + job[0], sr=44100)
            if len(y) is not 0:
                mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
                mfcc = mfcc.transpose()
                # For some samples the length is insufficient, just ignore them
                if len(mfcc) >= random_sample_size:
                    os.rename(job[1] + '/' + job[0], job[2] + '/' + job[0])
                    out_q.put([mfcc, job[3]])
                    in_q.task_done()
                    print('%s has been processed' % job[0])
        except NoBackendError:
            in_q.task_done()
        except EOFError:
            in_q.task_done()

def produce(in_q):
    for filename in os.listdir(music_files_path):
        print('into input queue: %s' % music_files_path + '/' + filename)
        in_q.put((filename, music_files_path, processed_music_files_path, [1., 0.]))

    for filename in os.listdir(nonmusic_files_path):
        print('into input queue: %s' % nonmusic_files_path + '/' + filename)
        in_q.put((filename, nonmusic_files_path, processed_nonmusic_files_path, [0., 1.]))

def main():

    in_q = mp.JoinableQueue()
    out_q = mp.Queue()

    produce(in_q)

    for _ in range(4):
        in_q.put(None)

    procs = []
    for _ in range(4):
        p = mp.Process(target=consume, args=(in_q, out_q,))
        p.start()
        procs.append(p)

    write_proc = mp.Process(target=persistance, args=(out_q,))
    write_proc.start()
    procs.append(write_proc)

    in_q.join()
    out_q.put(None)

    for p in procs:
        p.join()

def persistance(q):
    limit = 2000
    stop = False
    dump_list = []
    print('process queue!')
    while True:
        with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
            count = 0
            if stop:
                break
            while count < limit:
                #with open(base_url + '/eval_data.dat', 'wb') as fp:
                feature = q.get()
                print('get no.%g feature' % count)
                if feature is None:
                    print('get None feature, stop the process!')
                    stop = True
                    break
                if len(feature[0]) >= 256:
                    dump_list.append(feature)
                    count += 1
            dill.dump(dump_list, fp)
            print('%g files has been processed and dumped!' % count)
            del dump_list[:]

    if len(dump_list) is not 0:
        with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
            dill.dump(dump_list, fp)
            print('the last %g files has been processed and dumped!' % len(dump_list))

if __name__ == '__main__':
    main()
