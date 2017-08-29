import subprocess as sp
import os

music_folder = '/Users/rui.zhong/audio-recognition/AudioSet/music'
nonmusic_folder = '/Users/rui.zhong/audio-recognition/AudioSet/nonmusic'

def convert():
    for f in os.listdir(music_folder):
        input_file = music_folder + '/' + f
        output_file = input_file.replace('.wav', '.m4a')
        cmd = [ 'ffmpeg',
            '-i', input_file,
            '-n',
            '-f', 'mp4',
            '-ar', '44100',
            '-ac', '2',
            '-t', '10',
            '-threads', '0',
            output_file]
        proc = sp.Popen(cmd, stdout=sp.PIPE)
        outputs, err = proc.communicate()
        if proc.returncode == 0:
            os.remove(input_file)

    for f in os.listdir(nonmusic_folder):
        input_file = music_folder + '/' + f
        output_file = input_file.replace('.wav', '.m4a')
        cmd = [ 'ffmpeg',
            '-i', input_file,
            '-n',
            '-f', 'mp4',
            '-ar', '44100',
            '-ac', '2',
            '-t', '10',
            '-threads', '0',
            output_file]
        proc = sp.Popen(cmd, stdout=sp.PIPE)
        outputs, err = proc.communicate()
        if proc.returncode == 0:
            os.remove(input_file)


convert()
