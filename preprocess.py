import os
import librosa
import pickle
import dill
from datetime import datetime

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


def process_one_file(q):
    while True:
        job_ = q.get()
        job = dill.loads(job_)
        try:
            y, sr = librosa.load(job.filename, sr=44100)
            if len(y) is not 0:
                mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
                mfcc = mfcc.transpose()
                # For some samples the length is insufficient, just ignore them
                if len(mfcc) >= random_sample_size:
                    if job.is_music:
                        job.output_q.put([mfcc, [1., 0.]])
                    else:
                        job.output_q.put([mfcc, [0., 1.]])
                    os.rename(job.file_path + '/' + job.filename, job.processed_files_path + '/' + job.filename)
        except:
            pass


class Job(object):
    def __init__(self, filename, file_path, processed_files_path, is_music, output_q):
        self.filename = filename
        self.is_music = is_music
        self.file_path = file_path
        self.processed_files_path = processed_files_path
        self.output_q = output_q

def main():
    manager = mp.Manager()
    input_q = manager.Queue(1)
    output_q = manager.Queue(1)
    for filename in os.listdir(music_files_path):
        input_q.put(dill.dumps(filename, music_files_path, processed_music_files_path, True, output_q))

    for filename in os.listdir(nonmusic_files_path):
        input_q.put(dill.dumps(filename, nonmusic_files_path, processed_nonmusic_files_path, False, output_q))

    with mp.Pool(process=4) as pool:
        pool.apply_async(process_one_file, input_q)

    persistance(output_q)

def persistance(q):
    limit = 4000
    dump_list = []
    while not q.empty():
        count = 0
        while count < limit:
            with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
            #with open(base_url + '/eval_data.dat', 'wb') as fp:
                feature = q.get(False)
                if len(feature[0]) >= 256:
                    dump_list.append(feature)
                    count += 1
        dill.dump(dump_list, fp)

    with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
        if len(dump_list) is not 0:
            pickle.dump(dump_list, fp)

if __name__ == '__main__':
    main()
