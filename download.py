import csv
import random
import subprocess

youtube_url_prefix = 'http://youtu.be/'
music_label = '/m/04rlf'
music_file_path = '/home/centos/audio-recognition/AudioSet/music/'
nonmusic_file_path = '/home/centos/audio-recognition/AudioSet/nonmusic/'
data_csv_file = '/home/centos/audio-recognition/AudioSet/unbalanced_train_segments.csv'
#music_file_path = '/Users/rui.zhong/audio-recognition/AudioSet/eval_music/'
#nonmusic_file_path = '/Users/rui.zhong/audio-recognition/AudioSet/eval_nonmusic/'
#data_csv_file = '/Users/rui.zhong/audio-recognition/AudioSet/eval_segments.csv'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SoundClip(object):
    def __init__(self, youtube_id, start_time, end_time, labels):
        self.youtube_id = youtube_id
        self.start_time = start_time
        self.end_time = end_time
        self.labels = list(map(lambda x: x.replace('\"','').strip(), labels))
        if music_label in self.labels:
            self.is_music = True
            self.file_name = music_file_path + youtube_id + '.m4a'
        else:
            self.is_music = False
            self.file_name = nonmusic_file_path + youtube_id + '.m4a'
        self.url = youtube_url_prefix + youtube_id

def download_one_audio(sound_clip):
    try:
        gen_cmd = 'youtube-dl -x -g ' + sound_clip.url
        gen_proc = subprocess.Popen(gen_cmd.split(), stdout=subprocess.PIPE)
        outputs, err = gen_proc.communicate()
        if gen_proc.returncode is not 0:
            return False
        download_str = outputs.split()
        download_cmd = ['ffmpeg',
                 '-ss', sound_clip.start_time,
                 '-i', download_str[0],
                 '-n',
                 '-f', 'mp4',
                 '-ar', '44100',
                 '-ac', '2',
                 '-t', '10',
                 '-threads', '0',
                sound_clip.file_name]
        download_proc = subprocess.Popen(download_cmd, stdout=subprocess.PIPE)
        outputs, err = download_proc.communicate()
        if download_proc.returncode is 0:
            return True
        return False
    except Exception:
        # For some videos already deleted and some failed to encoding
        pass

def download_audio_set():
    with open(data_csv_file, mode='r') as source:
        reader = csv.reader(source)
        music_limit = 7000
        #music_limit = 500
        nonmusic_limit = 7000
        #nonmusic_limit = 500
        clip_list = []
        for row in reader:
            clip = SoundClip(row[0], row[1], row[2], row[3:])
            clip_list.append(clip)

        music_list = []
        nonmusic_list = []

        random.shuffle(clip_list)

        for clip in clip_list:
            if clip.is_music and len(music_list) < music_limit:
                if download_one_audio(clip):
                    print(bcolors.WARNING + 'True music! ' + clip.file_name + bcolors.ENDC)
                    music_list.append(clip)
            elif len(nonmusic_list) < nonmusic_limit:
                if download_one_audio(clip):
                    print(bcolors.WARNING + 'True nonmusic! ' + clip.file_name  + bcolors.ENDC)
                    nonmusic_list.append(clip)
            elif len(nonmusic_list) >= nonmusic_limit and len(music_list) >= music_limit:
                break;

        print('music count: %d, nonmusic count: %d' % (len(music_list), len(nonmusic_list)))

download_audio_set()
