
# pAura: Python AUdio Recording and Analysis

## News
 * [2020/01/02] paura2.py python3 support added 

## General
pAura is a Python tool for recording audio information and analyzing its content in an online and realtime manner.

paura.py uses alsaaudio to capture sound (to be used under Linux environments), while paura2.py uses pySound (based on portaudio), so it can both be used in Linux and MacOs environments.

## Installation (for paura.py and Linux)
 * Install [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/).
 * Install alsaaudio for python: `sudo apt-get install python-alsaaudio`
 * Install opencv for python: `sudo apt-get install python-opencv`
 * Clone the source of this library: 
 ```
git clone https://github.com/tyiannak/paura.git
```


## Installation (for paura2.py and MacOs)
 * Install [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/).
 * Install portaudio: `brew install portaudio`
 * Install pysound and opencv for python: `pip install pyaudio opencv-python`
 * Clone the source of this library: 
 ```
git clone https://github.com/tyiannak/paura.git
```

## Command line execution
The following command records audio data using blocks of 0.3sec. For each block, the spectrogram and chromagram are plotted, along with the raw samples (waveform). Also, a simple activity detection is performed and each non-silent segment is stored in a WAV file. Successive blocks are merged to a single WAV file.

```
python paura.py recordAndAnalyze --blocksize 0.3 --spectrogram --chromagram --recordactivity
```

```
python paura2.py recordAndAnalyze --blocksize 0.3 --spectrogram --chromagram --recordactivity
```


The sound detection method is very simple and requires a short "silence" period at the begining of the recording (10 blocks, 3 seconds for the above example).
Arguments --spectrogram, --chromagram and --recordactivity are optional

## Ongoing work
Online classification and clustering

## Author
<img src="http://users.iit.demokritos.gr/~tyianak/files/me.jpg" align="left" height="100"/>

[Theodoros Giannakopoulos] (http://www.di.uoa.gr/~tyiannak), 
Postdoc researcher at NCSR Demokritos, 
Athens,
Greece


