
# pAura: Python Audio Recorder and Analyzer

## General
pAura is a Python tool for recording audio information and analyzing its content in an online and realtime manner.

## Installation
 * Install dependencies: 
 * * Install [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/).
 * * Install alsaaudio for python: `sudo apt-get install python-alsaaudio`
 * * Install opencv for python: `sudo apt-get install python-opencv`
 * Clone the source of this library: 
 ```
git clone https://github.com/tyiannak/paura.git
```

## Command line execution
The following command records audio data using blocks of 300 mseconds (0.3). For each block, the spectrogram and chromagram are plotted, along with the raw signal. Also, a simple activity detection is performed and each detected segment is stored in a WAV file (successive blocks are merged to a single WAV file).

```
python paura.py recordAndAnalyze --blocksize 0.3 --spectrogram --chromagram --recordactivity
```

The sound detection method is very simple and requires a short "silence" period at the begining of the recording (10 blocks, 3 seconds for the above example)

## Author
<img src="http://cgi.di.uoa.gr/~tyiannak/image.jpg" align="left" height="100"/>

[Theodoros Giannakopoulos] (http://www.di.uoa.gr/~tyiannak), 
Postdoc researcher at NCSR Demokritos, 
Athens,
Greece


