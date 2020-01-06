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
import struct

global Fs
Fs = 16000
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
    wavfile.write("output.wav", Fs, numpy.int16(all_data))
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Real time audio analysis")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks",
        dest="task", metavar="")
    recordAndAnalyze = tasks.add_parser("recordAndAnalyze",
                                        help="Get audio data "
                                             "from mic and analyze")
    recordAndAnalyze.add_argument("-bs", "--blocksize",
                                  type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5,
                                                       1],
                                  default=0.20, help="Recording block size")
    recordAndAnalyze.add_argument("-fs", "--samplingrate", type=int,
                                  choices=[4000, 8000, 16000, 32000, 44100],
                                  default=16000, help="Recording block size")
    recordAndAnalyze.add_argument("--chromagram", action="store_true",
                                  help="Show chromagram")
    recordAndAnalyze.add_argument("--spectrogram", action="store_true",
                                  help="Show spectrogram")
    recordAndAnalyze.add_argument("--recordactivity", action="store_true",
                                  help="Record detected sounds to wavs")
    return parser.parse_args()


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


def recordAudioSegments(block_size, Fs=16000,
                        show_spec=False,
                        show_chroma=False,
                        rec_activity=False):
    mid_buf_size = int(Fs * block_size)

    print("Press Ctr+C to stop recording")

    start_time_str = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")

    # MEAN, STD = loadMEANS("svmMovies8classesMEANS")

    pa = pyaudio.PyAudio()

    stream = pa.open(format=FORMAT,
                     channels=1,
                     rate=Fs,
                     input=True,
                     frames_per_buffer=mid_buf_size)

    mid_buf = []
    cur_win = []
    count = 0
    global all_data
    all_data = []
    energy_100_buffer_zero = []
    cur_active_win = numpy.array([])
    time_start = time.time()

    while 1:
        try:
            block = stream.read(mid_buf_size)
            count_b = len(block) / 2
            format = "%dh" % (count_b)
            shorts = struct.unpack(format, block)
            cur_win = list(shorts)
            mid_buf = mid_buf + cur_win;  # copy to mid_buf
            del cur_win

            if 1:
                # time since recording started:
                e_time = (time.time() - time_start)
                # data-driven time
                data_time = (count + 1) * block_size
                wavfile.write("temp.wav", Fs, numpy.int16(mid_buf))
                print(len(mid_buf))
                flags, classes, _, _ = aS.mtFileClassification("temp.wav",
                                                               "model",
                                                               "svm",
                                                               False, "")
                print(classes[int(flags[-1])])

                all_data += mid_buf
                mid_buf = numpy.double(mid_buf)

                # Compute spectrogram
                if show_spec:
                    (spec, t_axis, freq_axis_s) = sF.spectrogram(mid_buf, 
                                                                 Fs, 
                                                                 0.020 * Fs, 
                                                                 0.02 * Fs)
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
                                                                    0.020 * Fs,
                                                                    0.02 * Fs)  
                    freq_axis_c = numpy.array(freq_axis_c)  
                    # most dominant chroma classes:
                    dominant_freqsC = freq_axis_c[numpy.argmax(chrom,
                                                               axis=1)]
                    # get most common among all short-term windows
                    max_freqC = most_common(dominant_freqsC)[0]

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
                energy_100 = (100 * numpy.sum(mid_buf * mid_buf)
                             / (mid_buf.shape[0] * 32000 * 32000))
                if count < 10:  # TODO make this param
                    energy_100_buffer_zero.append(energy_100)
                    mean_energy_100_zero = numpy.mean(
                        numpy.array(energy_100_buffer_zero))
                else:
                    mean_energy_100_zero = numpy.mean(
                        numpy.array(energy_100_buffer_zero))
                    if (energy_100 < 1.2 * mean_energy_100_zero):
                        # if a sound has been detected in the previous segment:
                        if cur_active_win.shape[0] > 0:
                            # set time of current active window
                            active_t2 = e_time - block_size
                            if active_t2 - active_t1 > min_act_dur:
                                wav_fname = start_time_str + \
                                              "_activity_{0:.2f}" \
                                              "_{1:.2f}.wav".format(active_t1,
                                                                    active_t2)
                                if rec_activity:
                                    # write current active window to file
                                    wavfile.write(wav_fname, Fs, numpy.int16(
                                        cur_active_win))
                            # delete current active window:
                            cur_active_win = numpy.array([])
                    else:
                        if cur_active_win.shape[0] == 0:
                            # if this is a new active window!
                            active_t1 = e_time - block_size
                            # set timestamp start of new active window
                        cur_active_win = numpy.concatenate(
                            (cur_active_win, mid_buf))
                        # Show status messages on Status cv winow:
                textIm = numpy.zeros((status_h, plot_w, 3))
                statusStrTime = "time: %.1f sec" % e_time + \
                                " - data time: %.1f sec" % data_time + \
                                " - loss : %.1f sec" % (e_time - data_time)
                statusStrFeature = "ene1:%.1f" % energy_100 + \
                                   " eneZero:%.1f" % mean_energy_100_zero
                cv2.putText(textIm, statusStrTime, (0, 11),
                            cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                cv2.putText(textIm, statusStrFeature, (0, 22),
                            cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                if cur_active_win.shape[0] > 0:
                    cv2.putText(textIm, "sound", (0, 33),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                else:
                    cv2.putText(textIm, "silence", (0, 33),
                                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 220))
                cv2.imshow("Status", textIm)
                cv2.moveWindow("Status", 50, 0)
                mid_buf = []
                ch = cv2.waitKey(10)
                count += 1
        except IOError:
            print("Error recording")


if __name__ == "__main__":
    args = parse_arguments()
    if args.task == "recordAndAnalyze":
        Fs = args.samplingrate
        recordAudioSegments(args.blocksize, args.samplingrate, args.spectrogram,
                            args.chromagram, args.recordactivity)
