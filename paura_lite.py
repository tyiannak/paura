import numpy as np
import pyaudio
import struct
import scipy.fftpack as scp
import termplotlib as tpl
import os

# get window's dimensions
rows, columns = os.popen('stty size', 'r').read().split()

buff_size = 0.2   # window size in seconds
wanted_num_of_bins = 40  # number of frequency bins to display

# initialize soundcard for recording:
fs = 8000
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs,
                 input=True, frames_per_buffer=int(fs * buff_size))
while 1:
    # get current block and convert to list of short ints,
    block = stream.read(int(fs * buff_size))
    format = "%dh" % (len(block) / 2)
    shorts = struct.unpack(format, block)
    # then normalize and convert to numpy array:
    x = np.double(list(shorts)) / (2**15)
    seg_len = len(x)
    energy = np.mean(x ** 2)
    max_energy = 0.02
    max_width_from_energy = int((energy / max_energy) * int(columns)) + 1
    if max_width_from_energy > int(columns) - 10:
        max_width_from_energy = int(columns) - 10
    # get the magnitude of the fft and the corresponding frequencies
    X = np.abs(scp.fft(x))[0:int(seg_len/2)]
    freqs = (np.arange(0, 1 + 1.0/len(X), 1.0 / len(X)) * fs / 2)
    # and resample to a fix num of frequency bins (to visualize)
    wanted_step = (int(freqs.shape[0] / wanted_num_of_bins))
    freqs2 = freqs[0::wanted_step].astype('int')
    X2 = np.mean(X.reshape(-1, wanted_step), axis=1)
    # plot (freqs, fft) as horizontal histogram
    fig = tpl.figure()
    fig.barh(X2, labels=[str(int(f)) + " Hz" for f in freqs2[0:-1]],
             show_vals=False, max_width=max_width_from_energy)
    fig.show()
    print("\n" * (int(rows) - freqs2.shape[0] - 1))