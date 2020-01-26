
# pAura: Python AUdio Recording and Analysis

## News
 * [2020/01/02] paura2.py python3 support added 

## General
pAura is a Python tool for recording audio information and analyzing its content
 in an online and realtime manner.

`paura.py` uses pySound (based on portaudio), so it can both be used in Linux 
and MacOs environments.

(Old code, noa available in paura_old_als.py uses alsaaudio to capture sound 
 and is therefore to be used only under Linux environments)

## Installation
Before downloading this library and setting up the pip requirements, please 
consider the following:

### Requirements for Linux
 * Install [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/).
 * Install portaudio: `sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0`
 * Install opencv for python: `sudo apt-get install python-opencv`

### Requirements for MacOs
 * Install [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/).
 * Install portaudio: `brew install portaudio`
 * Install pysound and opencv for python: `pip3 install pyaudio opencv-python`


## Execution and outputs

### Execution example
The following command records audio data using blocks of 1sec (segments). 

```
 python3 paura2.py --blocksize 1.0 --spectrogram --chromagram --record_segments --record_all
```

### Outputs
For each segment, the script:
1) Visualizes the spectrogram, chromagram  along with the raw samples (waveform)
2) Applies a simple audio classifier that distinguishes between 4 classes namely
 silence, speech, music and other sounds.

### Outputs formats

The predictions are printed in the console in the form of timestamp 
(segment starting point in seconds, counting from the recording starting time) 
and class label (silence, music, speech or other), e.g.
```
1.65 silence
2.68 other
3.70 silence
4.72 other
5.65 speech
6.67 other
...
```

Also, the waveform, spectrograms and chromagrams are visualized in dedicated 
plots. 

If the `--record_segments` flag is provided, 
each segment is saved in a folder named by the starting timestamp of the 
recording session, and has a filename indicated by its relative timestamp from 
the recording starting time, e.g. for the above example:
```
⇒ ls -1 2020_01_26_11:16PM_segments 
0001.65_silence.wav
0002.68_other.wav
0003.70_silence.wav
0004.72_other.wav
0005.65_speech.wav
0006.67_other.wav
...
```

Finally, if `--record_all` is provided, the whole recording is saved in a 
singe audio file. Not to be used for very long recordings (many hours), due to 
memory issues. In the above example, the overall audio recording is stored in 
`2020_01_26_11:16PM.wav`


## Ongoing work
Merge Linux and MacOs in a single file

## Author
<img src="https://tyiannak.github.io/files/3.JPG" align="left" height="100"/>

[Theodoros Giannakopoulos](https://tyiannak.github.io),
Director of Machine Learning at [Behavioral Signals](https://behavioralsignals.com)


