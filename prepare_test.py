import os
import csv
import pickle
import librosa

folder = '/home/centos/audio-recognition/test/test_medium/'
csv_file = folder + 'test.csv'

random_sample_size = 256

proc_list = []

with open(csv_file, 'r') as cf:
    reader = csv.reader(cf)
    for row in reader:
        filename = row[0] + '.mp4'
        print('start process %s' % filename)
        is_music = int(row[1])
        y, sr = librosa.load(folder + filename, sr=44100)
        if len(y) is not 0:
            mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
            mfcc = mfcc.transpose()
            # For some samples the length is insufficient, just ignore them
            if len(mfcc) >= random_sample_size:
                print('file size of %s is %g * 64' % (filename, len(mfcc)))
                if is_music == 1:
                    proc_list.append([mfcc, [0., 1.]])
                else:
                    proc_list.append([mfcc, [1., 0.]])

print(len(proc_list))

with open(folder + 'eval.prod.dat', 'wb') as fp:
    pickle.dump(proc_list, fp)

print('finish!')
