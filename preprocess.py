import os
import librosa
import unittest
import pickle
from datetime import datetime

class AudioSetPreprocess(object):
    def __init__(self):
        self.random_sample_size = 256
        self.base_url = '/home/centos/audio-recognition/AudioSet'
        self.music_files_limit = 15000
        self.nonmusic_files_limit = 15000
        self.music_files_path = self.base_url + '/music'
        self.processed_music_files_path = self.base_url + '/processed/music'
        self.nonmusic_files_path = self.base_url + '/nonmusic'
        self.processed_nonmusic_files_path = self.base_url + '/processed/nonmusic'
        #self.music_files_limit = 100
        #self.nonmusic_files_limit = 100
        #self.music_files_path = self.base_url + '/eval_music'
        #self.processed_music_files_path = self.base_url + '/processed/eval_music'
        #self.nonmusic_files_path = self.base_url + '/eval_nonmusic'
        #self.processed_nonmusic_files_path = self.base_url + '/processed/eval_nonmusic'

    def process_one_file(self, filename, class_list):
        y, sr = librosa.load(filename, sr=44100)
        if len(y) is 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
        mfcc = mfcc.transpose()
        print(filename)
        # For some samples the length is insufficient, just ignore them
        if len(mfcc) < self.random_sample_size:
            return None
        return [mfcc, class_list]

    def preprocess_batch(self, files_dir, limit, class_list, processed_dir):
        processed_list = []
        count = 0
        for filename in os.listdir(files_dir):
            if count >= limit:
                break
            processed = self.process_one_file(files_dir + '/' + filename, class_list)
            os.rename(files_dir + '/' + filename, processed_dir + '/' + filename)
            if processed is not None:
                processed_list.append(processed)
                count = count + 1

        return processed_list

    def preprocess(self):
        processed_list = []
        processed_list.extend(self.preprocess_batch(
          files_dir=self.music_files_path,
          limit=self.music_files_limit,
          class_list=[1., 0.],
          processed_dir=self.processed_music_files_path))
        processed_list.extend(self.preprocess_batch(
          files_dir=self.nonmusic_files_path,
          limit=self.nonmusic_files_limit,
          class_list=[0., 1.],
          processed_dir=self.processed_nonmusic_files_path))
        return processed_list

    def persistance(self):
        librosa.cache.clear()
        processed_list = self.preprocess()
        print(len(processed_list))
        with open(self.base_url + '/data.' + datetime.now().strftime('%s'), 'wb') as fp:
        #with open(self.base_url + '/eval_data.dat', 'wb') as fp:
            pickle.dump(processed_list, fp)
            librosa.cache.clear()

class PreprocessTest(unittest.TestCase):

    def __init__(self):
        self.preprocess = AudioSetPreprocess()
        self.test_files_dir = '/Users/rui.zhong/audio-recognition/test'
        self.expected_list = pickle.load(open(self.test_files_dir + '/preprocess.result', 'rb'))

    def test(self):
        test_file = self.preprocess.process_one_file(self.test_files_dir + '/test_wav_file_01.wav', [1., 0.])
        self.assertListEqual(test_file[0], self.expected_list)

AudioSetPreprocess().persistance()
