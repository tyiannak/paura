import numpy as np
import pyaudio
import struct
import scipy.fftpack as scp
import termplotlib as tpl
import os

rows, columns = os.popen('stty size', 'r').read().split()

buff_size = 0.2
wanted_num_of_bins = 40

# initialize soundcard for recording:
fs = 8000
mid_buf_size = int(fs * buff_size)
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs,
                 input=True, frames_per_buffer=mid_buf_size)
while 1:
    # get current block and convert to list of short ints,
    block = stream.read(mid_buf_size)
    format = "%dh" % (len(block) / 2)
    shorts = struct.unpack(format, block)
    # then to numpy array:
    x = np.double(list(shorts)) / (2**15)
    seg_len = len(x)
    # get the magnitude of the fft and the corresponding frequencies
    # (and resample to visualize)
    X = np.abs(scp.fft(x))[0:int(seg_len/2)]
    freqs = (np.arange(0, 1 + 1.0/len(X), 1.0 / len(X)) * fs / 2)
    wanted_step = (int(freqs.shape[0] / wanted_num_of_bins))
    freqs2 = freqs[0::wanted_step].astype('int')
    X2 = np.mean(X.reshape(-1, wanted_step), axis=1)
    # plot (freqs, fft) as horizontal histogram
    fig = tpl.figure()
    fig.barh(X2, labels=[str(int(f)) + " Hz" for f in freqs2[0:-1]],
             show_vals=False, max_width=int(columns) - 20)
    fig.show()
    print("\n" * (int(rows) - freqs2.shape[0] - 1))