import sys, time, numpy, scipy, cv2
import argparse
import scipy.io.wavfile as wavfile
from pyAudioAnalysis import ShortTermFeatures as sF
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
allData = []
HeightPlot = 150
WidthPlot = 720
statusHeight = 150;
minActivityDuration = 1.0


def signal_handler(signal, frame):
    """
    This function is called when Ctr + C is pressed and is used to output the
    final buffer into a WAV file
    """
    # write final buffer to wav file
    wavfile.write("output.wav", Fs, numpy.int16(allData))
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
                                  type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5],
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


def plotCV(Fun, Width, Height, MAX):
    if len(Fun) > Width:
        hist_item = Height * (Fun[len(Fun) - Width - 1:-1] / MAX)
    else:
        hist_item = Height * (Fun / MAX)
    h = numpy.zeros((Height, Width, 3))
    hist = numpy.int32(numpy.around(hist_item))

    for x, y in enumerate(hist):
        cv2.line(h, (x, int(Height / 2)),
                 (x, Height - y), (255, 0, 255))

    return h


"""
Core functionality:
"""


def recordAudioSegments(BLOCKSIZE, Fs=16000,
                        showSpectrogram=False,
                        showChromagram=False,
                        recordActivity=False):
    midTermBufferSize = int(Fs * BLOCKSIZE)

    print("Press Ctr+C to stop recording")

    startDateTimeStr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")

    # MEAN, STD = loadMEANS("svmMovies8classesMEANS")

    pa = pyaudio.PyAudio()

    stream = pa.open(format=FORMAT,
                     channels=1,
                     rate=Fs,
                     input=True,
                     frames_per_buffer=midTermBufferSize)

    midTermBuffer = []
    curWindow = []
    count = 0
    global allData
    allData = []
    energy100_buffer_zero = []
    curActiveWindow = numpy.array([])
    timeStart = time.time()

    while 1:
        try:
            block = stream.read(midTermBufferSize)
            countB = len(block) / 2
            format = "%dh" % (countB)
            shorts = struct.unpack(format, block)
            curWindow = list(shorts)
            midTermBuffer = midTermBuffer + curWindow;  # copy to midTermBuffer
            del (curWindow)
            if 1:
                # time since recording started:
                elapsedTime = (time.time() - timeStart)
                # data-driven time
                dataTime = (count + 1) * BLOCKSIZE

                # TODO
                # mtF, _ = aF.mtFeatureExtraction(midTermBuffer, Fs, BLOCKSIZE * Fs, BLOCKSIZE * Fs, 0.050 * Fs, 0.050 * Fs)
                # curFV = (mtF - MEAN) / STD
                # TODO
                allData += midTermBuffer
                midTermBuffer = numpy.double(midTermBuffer)

                # Compute spectrogram
                if showSpectrogram:
                    (spectrogram, TimeAxisS, FreqAxisS) = sF.spectrogram(
                        midTermBuffer, Fs, 0.020 * Fs,
                        0.02 * Fs)  # extract spectrogram
                    FreqAxisS = numpy.array(FreqAxisS)  # frequency axis
                    # most dominant frequencies (for each short-term window):
                    DominantFreqs = FreqAxisS[numpy.argmax(spectrogram,
                                                           axis=1)]
                    maxFreq = numpy.mean(
                        DominantFreqs)  # get average most dominant freq
                    maxFreqStd = numpy.std(DominantFreqs)
                    # Compute chromagram                        
                if showChromagram:
                    (chromagram, TimeAxisC, FreqAxisC) = sF.chromagram(
                        midTermBuffer, Fs, 0.020 * Fs,
                        0.02 * Fs)  # get chromagram
                    FreqAxisC = numpy.array(
                        FreqAxisC)  # frequency axis (12 chroma classes)
                    # most dominant chroma classes:
                    DominantFreqsC = FreqAxisC[numpy.argmax(chromagram,
                                                            axis=1)]
                    maxFreqC = most_common(DominantFreqsC)[
                        0]  # get most common among all short-term windows

                # Plot signal window
                signalPlotCV = plotCV(
                    scipy.signal.resample(midTermBuffer + 16000, WidthPlot),
                    WidthPlot, HeightPlot, 32000)
                cv2.imshow('Signal', signalPlotCV)
                cv2.moveWindow('Signal', 50, statusHeight + 50)
                # Show spectrogram
                if showSpectrogram:
                    iSpec = numpy.array(spectrogram.T * 255, dtype=numpy.uint8)
                    iSpec2 = cv2.resize(iSpec, (WidthPlot, HeightPlot),
                                        interpolation=cv2.INTER_CUBIC)
                    iSpec2 = cv2.applyColorMap(iSpec2, cv2.COLORMAP_JET)
                    cv2.putText(iSpec2, "maxFreq: %.0f Hz" % maxFreq, (0, 11),
                                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                    cv2.imshow('Spectrogram', iSpec2)
                    cv2.moveWindow('Spectrogram', 50,
                                   HeightPlot + statusHeight + 60)
                # Show chromagram
                if showChromagram:
                    iChroma = numpy.array(
                        (chromagram.T / chromagram.max()) * 255,
                        dtype=numpy.uint8)
                    iChroma2 = cv2.resize(iChroma, (WidthPlot, HeightPlot),
                                          interpolation=cv2.INTER_CUBIC)
                    iChroma2 = cv2.applyColorMap(iChroma2, cv2.COLORMAP_JET)
                    cv2.putText(iChroma2, "maxFreqC: %s" % maxFreqC, (0, 11),
                                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                    cv2.imshow('Chroma', iChroma2)
                    cv2.moveWindow('Chroma', 50,
                                   2 * HeightPlot + statusHeight + 60)
                # Activity Detection:
                energy100 = (100 * numpy.sum(midTermBuffer * midTermBuffer)
                             / (midTermBuffer.shape[0] * 32000 * 32000))
                if count < 10:  # TODO make this param
                    energy100_buffer_zero.append(energy100)
                    mean_energy100_zero = numpy.mean(
                        numpy.array(energy100_buffer_zero))
                else:
                    mean_energy100_zero = numpy.mean(
                        numpy.array(energy100_buffer_zero))
                    if (energy100 < 1.2 * mean_energy100_zero):
                        # if a sound has been detected in the previous segment:
                        if curActiveWindow.shape[0] > 0:
                            # set time of current active window
                            activeT2 = elapsedTime - BLOCKSIZE
                            if activeT2 - activeT1 > minActivityDuration:
                                wavFileName = startDateTimeStr + \
                                              "_activity_{0:.2f}" \
                                              "_{1:.2f}.wav".format(activeT1,
                                                                    activeT2)
                                if recordActivity:
                                    # write current active window to file
                                    wavfile.write(wavFileName, Fs, numpy.int16(
                                        curActiveWindow))
                            # delete current active window:
                            curActiveWindow = numpy.array([])
                    else:
                        if curActiveWindow.shape[0] == 0:
                            # if this is a new active window!
                            activeT1 = elapsedTime - BLOCKSIZE
                            # set timestamp start of new active window
                        curActiveWindow = numpy.concatenate(
                            (curActiveWindow, midTermBuffer))
                        # Show status messages on Status cv winow:
                textIm = numpy.zeros((statusHeight, WidthPlot, 3))
                statusStrTime = "time: %.1f sec" % elapsedTime + \
                                " - data time: %.1f sec" % dataTime + \
                                " - loss : %.1f sec" % (elapsedTime - dataTime)
                statusStrFeature = "ene1:%.1f" % energy100 + \
                                   " eneZero:%.1f" % mean_energy100_zero
                cv2.putText(textIm, statusStrTime, (0, 11),
                            cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                cv2.putText(textIm, statusStrFeature, (0, 22),
                            cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                if curActiveWindow.shape[0] > 0:
                    cv2.putText(textIm, "sound", (0, 33),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                else:
                    cv2.putText(textIm, "silence", (0, 33),
                                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 220))
                cv2.imshow("Status", textIm)
                cv2.moveWindow("Status", 50, 0)
                midTermBuffer = []
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
