import os
import librosa
import pickle
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
        job = q.get()
        try:
            y, sr = librosa.load(job[1] + '/' + job[0], sr=44100)
            if len(y) is not 0:
                mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
                mfcc = mfcc.transpose()
                # For some samples the length is insufficient, just ignore them
                if len(mfcc) >= random_sample_size:
                    job[4].put([mfcc, job[3]])
                    os.rename(job[1] + '/' + job[0], job[2] + '/' + job[0])
        except:
            pass

def main():
    manager = mp.Manager()
    input_q = manager.Queue(1)
    output_q = manager.Queue(1)
    for filename in os.listdir(music_files_path):
        input_q.put((filename, music_files_path, processed_music_files_path, [1., 0.], output_q))

    for filename in os.listdir(nonmusic_files_path):
        input_q.put((filename, nonmusic_files_path, processed_nonmusic_files_path, [0., 1.], output_q))

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
        pickle.dump(dump_list, fp)

    with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
        if len(dump_list) is not 0:
            pickle.dump(dump_list, fp)

if __name__ == '__main__':
    main()
