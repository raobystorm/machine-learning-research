import os
import librosa
import pickle
import numpy as np
import dill
from datetime import datetime
import time

import asyncio

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

async def consume(in_q, out_q):
    while True:
        job = await in_q.get()
        print('process %s' % job[0])
        y, sr = librosa.load(job[0], sr=44100)
        y = np.random.rand(800, 64)
        if len(y) is not 0:
            mfcc = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=64, n_fft=1102, hop_length=441, power=2.0, n_mels=64)
            #mfcc = np.random.rand(512, 64)
            mfcc = mfcc.transpose()
            # For some samples the length is insufficient, just ignore them
            if len(mfcc) >= random_sample_size:
                os.rename(job[1] + '/' + job[0], job[2] + '/' + job[0])
                await out_q.put([mfcc, job[3]])
                print('%s has been processed' % job[0])

"""
class Job(object):
    def __init__(self, filename, file_path, processed_files_path, is_music):
        self.filename = filename
        self.is_music = is_music
        self.file_path = file_path
        self.processed_files_path = processed_files_path
"""

async def produce(in_q):
    for filename in os.listdir(music_files_path):
        print('into input queue: %s' % music_files_path + '/' + filename)
        await in_q.put((filename, music_files_path, processed_music_files_path, [1., 0.]))

    for filename in os.listdir(nonmusic_files_path):
        print('into input queue: %s' % nonmusic_files_path + '/' + filename)
        await in_q.put((filename, nonmusic_files_path, processed_nonmusic_files_path, [0., 1.]))

async def run():

    in_q = asyncio.LifoQueue()
    out_q = asyncio.LifoQueue()

    consumer = asyncio.ensure_future(consume(in_q, out_q))

    await produce(in_q)
    await in_q.join()
    consumer.cancel()
    persistance(out_q)

def persistance(q):
    limit = 4000
    dump_list = []
    stop = False
    print('process queue!')
    while not q.empty():
        with open(base_url + '/data.clip.' + datetime.now().strftime('%s'), 'wb') as fp:
            count = 0
            if stop:
                break
            while count < limit:
                #with open(base_url + '/eval_data.dat', 'wb') as fp:
                feature = await q.get()
                print('get no.%g feature' % count)
                if feature is None:
                    stop = True
                    break
                if len(feature[0]) >= 256:
                    dump_list.append(feature)
                    count += 1
            dill.dump(dump_list, fp)


loop = asyncio.get_event_loop()
loop.run_until_complete(run())
loop.close()
