
# pAura: Python AUdio Recording and Analysis

## News
 * [2020-06-05] Published medium article [Basic Audio Handling: How to handle and process audio files in command-line and through basic Python programming](https://medium.com/behavioral-signals-ai/basic-audio-handling-d4cc9c70d64d). Please refer to this as introductory material for handling audio data.
 * [2020/06/01] `paura_lite.py` added: a very simple command-line recorder and real-time visualization
 * [2020/01/02] python3 support added  

## General
 - ```paura.py``` is a Python tool for recording and analyzing sounds in an online 
and continuous manner. `paura` uses pySound (based on portaudio), so it can 
both be used in Linux and MacOs environments. 
 - `paura_lite.py` is a very simple command-line recorder and real-time visualization
  
## Installation
Before downloading this library and setting up the pip requirements, please 
consider the following:

### Requirements for Linux
 * Install [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/).
 * Install portaudio: `sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0`
 * Install opencv for python: `sudo apt-get install python-opencv`
 * sudo apt-get install gnuplot (required only for `paura_lite.py`)

### Requirements for MacOs
 * Install [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/).
 * Install portaudio: `brew install portaudio`
 * Install pysound and opencv for python: `pip3 install pyaudio opencv-python`
 * brew install gnuplot (required only for `paura_lite.py`)

## Execution and outputs for `paura.py`

### Execution example
The following command records audio data using blocks of 1sec (segments). 

```
 python3 paura.py --blocksize 1.0 --spectrogram --chromagram --record_segments --record_all
```

### Output
For each segment, the script:
1) Visualizes the spectrogram, chromagram  along with the raw samples (waveform)
2) Applies a simple audio classifier that distinguishes between 4 classes namely
 silence, speech, music and other sounds.

### Output format

The predictions are printed in the console in the form of timestamp 
(segment starting point in seconds, counting from the recording starting time), 
class label (silence, music, speech or other), and prediction confidence, e.g.
```
...
12.71	other	0.52
13.63	speech	0.30
14.66	other	0.43
15.68	music	0.92
16.70	speech	0.30
...
```

Also, the waveform, spectrograms and chromagrams are visualized in dedicated 
plots. 

If the `--record_segments` flag is provided, 
each segment is saved in a folder named by the starting timestamp of the 
recording session, and has a filename indicated by its relative timestamp from 
the recording starting time, e.g. for the above example:
```
⇒ ls -1 2020_01_28_12:37AM_segments 
...
0012.71_other.wav
0013.63_speech.wav
0014.66_other.wav
0015.68_music.wav
0016.70_speech.wav
...
```

Finally, if `--record_all` is provided, the whole recording is saved in a 
singe audio file. Not to be used for very long recordings (many hours), due to 
memory issues. In the above example, the overall audio recording is stored in 
`2020_01_26_11:16PM.wav`

## Execution and outputs for `paura_lite.py`
This script takes no arguments and just records sounds, 
while visualizing each segment's spectrogram in the console.
```
python3 paura_lite.py
```

Sample output (for one of the recorded windows).
```
0 Hz     ███▊
100 Hz   █████████▉
200 Hz   █████▎
300 Hz   ███████▋
400 Hz   ███████████▌
500 Hz   ████▊
600 Hz   █████▍
700 Hz   ██████▉
800 Hz   ████
900 Hz   ██▋
1000 Hz  █████▍
1100 Hz  ███
1200 Hz  ███▉
1300 Hz  ██████
1400 Hz  ██████▌
1500 Hz  █████████
1600 Hz  ██████████████
1700 Hz  ██████▌
1800 Hz  ████
1900 Hz  █▋
2000 Hz  █▌
2100 Hz  ██▏
2200 Hz  ███▊
2300 Hz  ████████▎
2400 Hz  ████████████▌
2500 Hz  █████▉
2600 Hz  ████▊
2700 Hz  ██▉
2800 Hz  ██▏
2900 Hz  ███▎
3000 Hz  █████▎
3100 Hz  ███
3200 Hz  ██▎
3300 Hz  ██▊
3400 Hz  ███▋
3500 Hz  ██▏
3600 Hz  █▌
3700 Hz  █▊
3800 Hz  █▍
3900 Hz  █▋

```

A demo video of `paura_lite.py` is also available in the following video:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YEi9AmA-07s/0.jpg)](https://www.youtube.com/watch?v=YEi9AmA-07s)

## Ongoing work
Export selected features and mid-term representations

## Author
<img src="https://tyiannak.github.io/files/3.JPG" align="left" height="100"/>

[Theodoros Giannakopoulos](https://tyiannak.github.io),
Director of Machine Learning at [Behavioral Signals](https://behavioralsignals.com)


