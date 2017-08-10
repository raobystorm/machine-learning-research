import csv
import random
import subprocess

YOUTUBE_URL_PREFIX = 'http://youtu.be/'
MUSIC_LABEL = '/m/04rlf'
MUSIC_FILE_PATH = '/Users/rui.zhong/AudioSet/music/'
NONMUSIC_FILE_PATH = '/Users/rui.zhong/AudioSet/nonmusic/'
DATA_CSV_FILE = '/Users/rui.zhong/AudioSet/unbalanced_train_segments.csv'

class SoundClip(object):
  def __init__(self, youtube_id, start_time, end_time, labels):
    self.youtube_id = youtube_id
    self.start_time = start_time
    self.end_time = end_time
    self.labels = labels.split(',')
    if MUSIC_LABEL in labels:
      self.is_music = True
      self.file_name = MUSIC_FILE_PATH + youtube_id + '.wav'
    else:
      self.is_music = False
      self.file_name = NONMUSIC_FILE_PATH + youtube_id + '.wav'
    self.url = YOUTUBE_URL_PREFIX + youtube_id

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
             '-f', 'wav', 
             '-ar', '44100',
             '-ac', '2',
             '-t', '10',
            sound_clip.file_name]
    download_proc = subprocess.Popen(download_cmd, stdout=subprocess.PIPE)
    outputs, err = download_proc.communicate()
  except Exception:
    pass

def download_audio_set():
  with open(DATA_CSV_FILE, mode='r') as source:
    reader = csv.reader(source)
    music_count = 0
    nonmusic_cound = 0
    music_limit = 100
    nonmusic_limit = 100
    clip_list = []
    for row in reader:
      clip = SoundClip(row[0], row[1], row[2], row[3])
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