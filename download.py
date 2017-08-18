import csv
import random
import subprocess

youtube_url_prefix = 'http://youtu.be/'
music_label = '/m/04rlf'
#music_file_path = '/home/centos/audio-recognition/AudioSet/music/'
music_file_path = '/home/centos/audio-recognition/AudioSet/eval_music/'
#nonmusic_file_path = '/home/centos/audio-recognition/AudioSet/nonmusic/'
nonmusic_file_path = '/home/centos/audio-recognition/AudioSet/eval_nonmusic/'
#data_csv_file = '/home/centos/audio-recognition/AudioSet/balanced_train_segments.csv'
data_csv_file = '/home/centos/audio-recognition/AudioSet/eval_segments.csv'

class SoundClip(object):
    def __init__(self, youtube_id, start_time, end_time, labels):
        self.youtube_id = youtube_id
        self.start_time = start_time
        self.end_time = end_time
        self.labels = list(map(lambda x: x.replace('\"','').strip(), labels))
        if music_label in self.labels:
            self.is_music = True
            self.file_name = music_file_path + youtube_id + '.wav'
        else:
            self.is_music = False
            self.file_name = nonmusic_file_path + youtube_id + '.wav'
        self.url = youtube_url_prefix + youtube_id

def download_one_audio(sound_clip):
    try:
        gen_cmd = 'youtube-dl -x -g ' + sound_clip.url
        gen_proc = subprocess.Popen(gen_cmd.split(), stdout=subprocess.PIPE)
        outputs, err = gen_proc.communicate()
        download_str = outputs.split()
        print(sound_clip.file_name)
        download_cmd = ['ffmpeg',
                 '-ss', sound_clip.start_time,
                 '-i', download_str[0],
                 '-n',
                 '-f', 'wav',
                 '-ar', '44100',
                 '-ac', '2',
                 '-t', '10',
                sound_clip.file_name]
        download_proc = subprocess.Popen(download_cmd, stdout=subprocess.PIPE)
        outputs, err = download_proc.communicate()
    except Exception:
        # For some videos already deleted and some failed to encoding
        pass

def download_audio_set():
    with open(data_csv_file, mode='r') as source:
        reader = csv.reader(source)
        #music_limit = 6000
        music_limit = 500
        #nonmusic_limit = 14000
        nonmusic_limit = 500
        clip_list = []
        for row in reader:
            clip = SoundClip(row[0], row[1], row[2], row[3:])
            clip_list.append(clip)

        random.shuffle(clip_list)
        music_list = []
        nonmusic_list = []

        for clip in clip_list:
            if len(music_list) == music_limit and len(nonmusic_list) == nonmusic_limit:
                break

            if clip.is_music and len(music_list) < music_limit:
                music_list.append(clip)
            elif len(nonmusic_list) < nonmusic_limit:
                nonmusic_list.append(clip)

        for music in music_list:
            download_one_audio(music)

        for nonmusic in nonmusic_list:
            download_one_audio(nonmusic)

        print(len(music_list))
        print(len(nonmusic_list))

download_audio_set()
