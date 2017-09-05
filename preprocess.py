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


def process_one_file(filename, class_list, q):
    try:
        y, sr = librosa.load(filename, sr=44100)
        if len(y) is 0:
            return
        mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
        mfcc = mfcc.transpose()
        # For some samples the length is insufficient, just ignore them
        if len(mfcc) < random_sample_size:
            return
        q.put([mfcc, class_list])
    except:
        return


class Kicker(mp.Process):
    def start(self, queue):
        self.queue = queue

    def run(self):
        self.preprocess_batch(music_files_path, [1., 0.], processed_music_files_path, self.queue)
        self.preprocess_batch(nonmusic_files_path, [0., 1.], processed_nonmusic_files_path, self.queue)

    def preprocess_batch(files_dir, class_list, processed_dir, q):
        with mp.Pool(process=3) as pool:
            for filename in os.listdir(files_dir):
                pool.apply_async(process_one_file, (files_dir + '/' + filename, class_list,q))
                os.rename(files_dir + '/' + filename, processed_dir + '/' + filename)


def main():
    manager = mp.Manager()
    q = manager.Queue(1)
    kicker = Kicker()
    kicker.start(q)
    persistance(q)

def persistance(q):
    count = 0
    limit = 5000
    while True:
        count = 0
        dump_list = []
        while count < limit:
            with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
            #with open(base_url + '/eval_data.dat', 'wb') as fp:
                feature = q.get()
                if len(feature[0]) >= 256:
                    dump_list.append(feature)
                    count += 1
        pickle.dump(dump_list, fp)

if __name__ == '__main__':
    main()
