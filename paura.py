import sys, time, numpy, scipy, cv2
import argparse
import scipy.io.wavfile as wavfile
from pyAudioAnalysis import ShortTermFeatures as sF
from pyAudioAnalysis import audioSegmentation as aS
import scipy.signal
import itertools
import operator
import datetime
import signal
import pyaudio
import os
import struct
import shutil

global Fs
global all_data
global outstr
Fs = 8000
FORMAT = pyaudio.paInt16
all_data = []
plot_h = 150
plot_w = 720
status_h = 150;
min_act_dur = 1.0 # minimum duration of each activation


def signal_handler(signal, frame):
    """
    This function is called when Ctr + C is pressed and is used to output the
    final buffer into a WAV file
    """
    # write final buffer to wav file
    if len(all_data) > 1:
        wavfile.write(outstr + ".wav", Fs, numpy.int16(all_data))
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)



"""
Utility functions
"""


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


def plotCV(function, width, height, max_val):
    if len(function) > width:
        hist_item = height * (function[len(function) - width - 1:-1] / max_val)
    else:
        hist_item = height * (function / max_val)
    h = numpy.zeros((height, width, 3))
    hist = numpy.int32(numpy.around(hist_item))

    for x, y in enumerate(hist):
        cv2.line(h, (x, int(height / 2)),
                 (x, height - y), (255, 0, 255))

    return h


"""
Core functionality:
"""


def record_audio(block_size, Fs=8000, show_spec=False, show_chroma=False,
                 log_sounds=False, logs_all=False):

    mid_buf_size = int(Fs * block_size)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=Fs,
                     input=True, frames_per_buffer=mid_buf_size)
    mid_buf = []
    count = 0
    global all_data
    global outstr
    all_data = []
    time_start = time.time()
    outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")
    out_folder = outstr + "_segments"
    if log_sounds:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
    while 1:
        try:
            block = stream.read(mid_buf_size)
            count_b = len(block) / 2
            format = "%dh" % (count_b)
            shorts = struct.unpack(format, block)
            cur_win = list(shorts)
            mid_buf = mid_buf + cur_win
            del cur_win

            if 1:
                # time since recording started:
                e_time = (time.time() - time_start)
                # data-driven time
                data_time = (count + 1) * block_size
                wavfile.write("temp.wav", Fs, numpy.int16(mid_buf))
                flags, classes, _, _ = aS.mtFileClassification("temp.wav",
                                                               "model",
                                                               "svm",
                                                               False, "")
                current_class = classes[int(flags[-1])]
                if logs_all:
                    all_data += mid_buf
                mid_buf = numpy.double(mid_buf)

                # Compute spectrogram
                if show_spec:
                    (spec, t_axis, freq_axis_s) = sF.spectrogram(mid_buf, 
                                                                 Fs, 
                                                                 0.050 * Fs,
                                                                 0.050 * Fs)
                    freq_axis_s = numpy.array(freq_axis_s)  # frequency axis
                    # most dominant frequencies (for each short-term window):
                    dominant_freqs = freq_axis_s[numpy.argmax(spec, axis=1)]
                    # get average most dominant freq
                    max_freq = numpy.mean(dominant_freqs)
                    max_freq_std = numpy.std(dominant_freqs)

                # Compute chromagram                        
                if show_chroma:
                    (chrom, TimeAxisC, freq_axis_c) = sF.chromagram(mid_buf, 
                                                                    Fs, 
                                                                    0.050 * Fs,
                                                                    0.050 * Fs)
                    freq_axis_c = numpy.array(freq_axis_c)  
                    # most dominant chroma classes:
                    dominant_freqs_c = freq_axis_c[numpy.argmax(chrom,
                                                               axis=1)]
                    # get most common among all short-term windows
                    max_freqC = most_common(dominant_freqs_c)[0]

                # Plot signal window
                signalPlotCV = plotCV(scipy.signal.resample(mid_buf + 16000, 
                                                            plot_w),
                                      plot_w, plot_h, 32000)
                cv2.imshow('Signal', signalPlotCV)
                cv2.moveWindow('Signal', 50, status_h + 50)

                # Show spectrogram
                if show_spec:
                    i_spec = numpy.array(spec.T * 255, dtype=numpy.uint8)
                    i_spec2 = cv2.resize(i_spec, (plot_w, plot_h),
                                        interpolation=cv2.INTER_CUBIC)
                    i_spec2 = cv2.applyColorMap(i_spec2, cv2.COLORMAP_JET)
                    cv2.putText(i_spec2, "max_freq: %.0f Hz" % max_freq, 
                                (0, 11), cv2.FONT_HERSHEY_PLAIN, 
                                1, (200, 200, 200))
                    cv2.imshow('Spectrogram', i_spec2)
                    cv2.moveWindow('Spectrogram', 50,
                                   plot_h + status_h + 60)
                # Show chromagram
                if show_chroma:
                    i_chroma = numpy.array((chrom.T /
                                            chrom.max()) * 255,
                                           dtype=numpy.uint8)
                    i_chroma2 = cv2.resize(i_chroma, (plot_w, plot_h),
                                           interpolation=cv2.INTER_CUBIC)
                    i_chroma2 = cv2.applyColorMap(i_chroma2, cv2.COLORMAP_JET)
                    cv2.putText(i_chroma2, "max_freqC: %s" % max_freqC, (0, 11),
                                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                    cv2.imshow('Chroma', i_chroma2)
                    cv2.moveWindow('Chroma', 50,
                                   2 * plot_h + status_h + 60)

                # Activity Detection:
                print("{0:.2f}".format(e_time), current_class)
                if log_sounds:
                    # TODO: log audio files
                    out_file = os.path.join(out_folder,
                                            "{0:.2f}_".format(e_time).zfill(8) +
                                            current_class + ".wav")
                    shutil.copyfile("temp.wav", out_file)

                textIm = numpy.zeros((status_h, plot_w, 3))
                statusStrTime = "time: %.1f sec" % e_time + \
                                " - data time: %.1f sec" % data_time + \
                                " - loss : %.1f sec" % (e_time - data_time)
                cv2.putText(textIm, statusStrTime, (0, 11),
                            cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                cv2.putText(textIm, current_class, (0, 33),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                cv2.imshow("Status", textIm)
                cv2.moveWindow("Status", 50, 0)
                mid_buf = []
                ch = cv2.waitKey(10)
                count += 1
        except IOError:
            print("Error recording")


def parse_arguments():
    record_analyze = argparse.ArgumentParser(description="Real time "
                                                         "audio analysis")
    record_analyze.add_argument("-bs", "--blocksize",
                                  type=float, choices=[0.25, 0.5, 0.75, 1],
                                  default=1, help="Recording block size")
    record_analyze.add_argument("-fs", "--samplingrate", type=int,
                                  choices=[4000, 8000, 16000, 32000, 44100],
                                  default=8000, help="Recording block size")
    record_analyze.add_argument("--chromagram", action="store_true",
                                  help="Show chromagram")
    record_analyze.add_argument("--spectrogram", action="store_true",
                                  help="Show spectrogram")
    record_analyze.add_argument("--record_segments", action="store_true",
                                  help="Record detected sounds to wavs")
    record_analyze.add_argument("--record_all", action="store_true",
                                  help="Record the whole recording to a single"
                                       " audio file")
    return record_analyze.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    Fs = args.samplingrate
    if Fs != 8000:
        print("Warning! Segment classifiers have been trained on 8KHz samples. "
              "Therefore results will be not optimal. ")
    record_audio(args.blocksize, Fs, args.spectrogram,
                 args.chromagram, args.record_segments, args.record_all)
