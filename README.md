# Audio-recognition

## 必要なライブラリ

1, Tensorflow (1.2.1)<br>
2, Librosa (0.5.1)<br>
3, Youtube-dl (version 2017.08.06)<br>
4, FFmpeg (version N-86948-gbac508f)<br>

## Data Set
Google Audio Set https://research.google.com/audioset/<br>
その中のUnbalancedとBalancedデータセットを使える。<br>
Labelの中'Music'('/m/04rlf')があるビデオの音声を音楽として、他のが非音楽として二種類分けている。<br>

## Feature
LibrosaのMFCC特徴量を使っている。処理方法は
<a href='https://arxiv.org/pdf/1609.09430.pdf'>こちらの論文</a>を参照している<br>
