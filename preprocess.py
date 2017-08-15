import os
import importlib
import numpy as np
import random
import pickle
import gc
from datetime import datetime

random_sample_size = 128

base_url = '/home/centos/audio-recognition/AudioSet'

music_files_path = base_url + '/music'
nonmusic_files_path = base_url + '/nonmusic'

processed_music_files_path = base_url + '/processed/music'
processed_nonmusic_files_path = base_url + '/processed/nonmusic'

eval_music_files_path = base_url + '/eval_music'
eval_nonmusic_files_path = base_url + '/eval_nonmusic'

def process_one_file(lib, filename, is_music):
  y, sr = lib.load(filename, sr=44100)
  mfcc = lib.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
  #mfcc = np.random.rand(64, 1001)
  mfcc = mfcc.transpose()
  print(filename)
  # For some samples the length is insufficient, just ignore them
  if len(mfcc) < random_sample_size:
    return None
  if is_music:
    return [mfcc, [1, 0]]
  else:
    return [mfcc, [0, 1]]

def preprocess(lib):
  processed_list = []
  count = 0
  limit = 1500
  for filename in os.listdir(music_files_path):
    if count >= limit:
      break
    processed = process_one_file(lib, music_files_path + '/' + filename, True)
    os.rename(music_files_path + '/' + filename, processed_music_files_path+ '/' + filename)
    if processed is not None:
      processed_list.append(processed)
      count = count + 1
  count = 0
  for filename in os.listdir(nonmusic_files_path):
    if count >= limit:
      break
    processed = process_one_file(lib, nonmusic_files_path + '/' + filename, False)
    os.rename(nonmusic_files_path + '/' + filename, processed_nonmusic_files_path+ '/' + filename)
    if processed is not None:
      processed_list.append(processed)
      count = count + 1
  return processed_list

def persistance():
  import librosa
  librosa.cache.clear()
  librosa = importlib.reload(librosa)
  processed_list = preprocess(librosa)
  random.shuffle(processed_list)
  print(len(processed_list))
  with open(base_url + '/data.' + datetime.now().strftime('%s'), 'wb') as fp:
  #with open(base_url + '/eval_data.dat', 'wb') as fp:
    pickle.dump(processed_list, fp)
    del processed_list
    gc.collect()

persistance()
