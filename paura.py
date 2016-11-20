import sys, os, alsaaudio, time, audioop, numpy, glob,  scipy, subprocess, wave, cPickle, threading, shutil, cv2
import argparse
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fftpack import rfft
from pyAudioAnalysis import audioFeatureExtraction as aF    
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from scipy.fftpack import fft
import matplotlib
import scipy.signal
import itertools
import operator
import datetime
import signal

allData = []
Fs = 16000
HeightPlot = 150  
WidthPlot = 720
statusHeight = 150;
minActivityDuration = 1.0

def signal_handler(signal, frame):
    wavfile.write("output.wav", Fs, numpy.int16(allData))  # write final buffer to wav file
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real time audio analysis")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar="")

    recordAndAnalyze = tasks.add_parser("recordAndAnalyze", help="Get audio data from mic and analyze")
    recordAndAnalyze.add_argument("-bs", "--blocksize", type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5], default=0.20, help="Recording block size")
    recordAndAnalyze.add_argument("--chromagram", action="store_true", help="Show chromagram")
    recordAndAnalyze.add_argument("--spectrogram", action="store_true", help="Show spectrogram")
    recordAndAnalyze.add_argument("--recordactivity", action="store_true", help="Record detected sounds to wavs")
    return parser.parse_args()

'''
Utitlity functions:
'''

def loadMEANS(modelName):
    # load pyAudioAnalysis classifier file (MEAN and STD values). 
    # used for feature normalization
    try:
        fo = open(modelName, "rb")
    except IOError:
            print "Load Model: Didn't find file"
            return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
    except:
        fo.close()
    fo.close()        
    return (MEAN, STD)

def most_common(L):    
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

def plotCV(Fun, Width, Height, MAX):
    if len(Fun)>Width:
        hist_item = Height * (Fun[len(Fun)-Width-1:-1] / MAX)
    else:
        hist_item = Height * (Fun / MAX)
    h = numpy.zeros((Height, Width, 3))
    hist = numpy.int32(numpy.around(hist_item))

    for x,y in enumerate(hist):        
            cv2.line(h,(x,Height/2),(x,Height-y),(255,0,255))        

    return h

'''
Basic functionality:
'''
def recordAudioSegments(BLOCKSIZE, showSpectrogram = False, showChromagram = False, recordActivity = False):    
    
    print "Press Ctr+C to stop recording"

    startDateTimeStr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")

    MEAN, STD = loadMEANS("svmMovies8classesMEANS")                                             # load MEAN feature values 

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK)                           # open alsaaudio capture 
    inp.setchannels(1)                                                                          # 1 channel
    inp.setrate(Fs)                                                                             # set sampling freq
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)                                                  # set 2-byte sample
    inp.setperiodsize(512)
    midTermBufferSize = int(Fs*BLOCKSIZE)
    midTermBuffer = []
    curWindow = []    
    count = 0
    global allData
    allData = []
    energy100_buffer_zero = []
    curActiveWindow = numpy.array([])    
    timeStart = time.time()

    while 1:            
            l,data = inp.read()                                                                 # read data from buffer
            if l:
                for i in range(len(data)/2):
                    curWindow.append(audioop.getsample(data, 2, i))                             # get audio samples
            
                if (len(curWindow)+len(midTermBuffer)>midTermBufferSize):
                    samplesToCopyToMidBuffer = midTermBufferSize - len(midTermBuffer)
                else:
                    samplesToCopyToMidBuffer = len(curWindow)

                midTermBuffer = midTermBuffer + curWindow[0:samplesToCopyToMidBuffer];          # copy to midTermBuffer
                del(curWindow[0:samplesToCopyToMidBuffer])


                if len(midTermBuffer) == midTermBufferSize:                                     # if midTermBuffer is full:
                    elapsedTime = (time.time() - timeStart)                                     # time since recording started
                    dataTime  = (count+1) * BLOCKSIZE                                           # data-driven time

                    # TODO
                    # mtF, _ = aF.mtFeatureExtraction(midTermBuffer, Fs, BLOCKSIZE * Fs, BLOCKSIZE * Fs, 0.050 * Fs, 0.050 * Fs)                    
                    # curFV = (mtF - MEAN) / STD
                    # TODO
                    allData += midTermBuffer                    
                    midTermBuffer = numpy.double(midTermBuffer)                                 # convert current buffer to numpy array                    

                    # Compute spectrogram
                    if showSpectrogram:                                                         
                        (spectrogram, TimeAxisS, FreqAxisS) = aF.stSpectogram(midTermBuffer, Fs, 0.020 * Fs, 0.02 * Fs) # extract spectrogram
                        FreqAxisS = numpy.array(FreqAxisS)                                      # frequency axis
                        DominantFreqs = FreqAxisS[numpy.argmax(spectrogram, axis = 1)]          # most dominant frequencies (for each short-term window)
                        maxFreq     = numpy.mean(DominantFreqs)                                 # get average most dominant freq
                        maxFreqStd  = numpy.std(DominantFreqs)                        
                    
                    # Compute chromagram                        
                    if showChromagram:                                                          
                        (chromagram, TimeAxisC, FreqAxisC) = aF.stChromagram(midTermBuffer, Fs, 0.020 * Fs, 0.02 * Fs)  # get chromagram
                        FreqAxisC = numpy.array(FreqAxisC)                                      # frequency axis (12 chroma classes)
                        DominantFreqsC = FreqAxisC[numpy.argmax(chromagram, axis = 1)]          # most dominant chroma classes 
                        maxFreqC = most_common(DominantFreqsC)[0]                               # get most common among all short-term windows

                    # Plot signal window
                    signalPlotCV = plotCV(scipy.signal.resample(midTermBuffer + 16000, WidthPlot), WidthPlot, HeightPlot, 32000)
                    cv2.imshow('Signal', signalPlotCV)
                    cv2.moveWindow('Signal',  50, statusHeight + 50)                    

                    # Show spectrogram
                    if showSpectrogram:
                        iSpec  = numpy.array(spectrogram.T * 255, dtype = numpy.uint8)
                        iSpec2 = cv2.resize(iSpec,(WidthPlot, HeightPlot), interpolation = cv2.INTER_CUBIC)
                        iSpec2 = cv2.applyColorMap(iSpec2, cv2.COLORMAP_JET)                    
                        cv2.putText(iSpec2, "maxFreq: %.0f Hz" % maxFreq, (0, 11), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200))
                        cv2.imshow('Spectrogram', iSpec2)  
                        cv2.moveWindow('Spectrogram',  50, HeightPlot + statusHeight + 60)
                    
                    # Show chromagram
                    if showChromagram:
                        iChroma  = numpy.array((chromagram.T / chromagram.max()) * 255, dtype = numpy.uint8)                
                        iChroma2 = cv2.resize(iChroma,(WidthPlot, HeightPlot), interpolation = cv2.INTER_CUBIC)
                        iChroma2 = cv2.applyColorMap(iChroma2, cv2.COLORMAP_JET)
                        cv2.putText(iChroma2, "maxFreqC: %s" % maxFreqC, (0, 11), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200))
                        cv2.imshow('Chroma', iChroma2)
                        cv2.moveWindow('Chroma',  50, 2 * HeightPlot + statusHeight + 60)

                    # Activity Detection:                    
                    energy100 = (100*numpy.sum(midTermBuffer * midTermBuffer) 
                        / (midTermBuffer.shape[0] * 32000 * 32000))     
                    if count < 10:                                                          # TODO make this param
                        energy100_buffer_zero.append(energy100)                    
                        mean_energy100_zero = numpy.mean(numpy.array(energy100_buffer_zero))
                    else:
                        mean_energy100_zero = numpy.mean(numpy.array(energy100_buffer_zero))
                        if (energy100 < 1.2 * mean_energy100_zero):
                            if curActiveWindow.shape[0] > 0:                                    # if a sound has been detected in the previous segment:
                                activeT2 = elapsedTime - BLOCKSIZE                              # set time of current active window
                                if activeT2 - activeT1 > minActivityDuration:
                                    wavFileName = startDateTimeStr + "_activity_{0:.2f}_{1:.2f}.wav".format(activeT1, activeT2)
                                    if recordActivity:
                                        wavfile.write(wavFileName, Fs, numpy.int16(curActiveWindow))# write current active window to file
                                curActiveWindow = numpy.array([])                               # delete current active window
                        else:
                            if curActiveWindow.shape[0] == 0:                                   # this is a new active window!
                                activeT1 = elapsedTime - BLOCKSIZE                              # set timestamp start of new active window
                            curActiveWindow = numpy.concatenate((curActiveWindow, midTermBuffer))                        

                    # Show status messages on Status cv winow:
                    textIm = numpy.zeros((statusHeight, WidthPlot, 3))
                    statusStrTime = "time: %.1f sec" % elapsedTime + " - data time: %.1f sec" % dataTime + " - loss : %.1f sec" % (elapsedTime-dataTime)                                        
                    statusStrFeature = "ene1:%.1f" % energy100 + " eneZero:%.1f"%mean_energy100_zero 
                    cv2.putText(textIm, statusStrTime, (0, 11),  cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200))
                    cv2.putText(textIm, statusStrFeature, (0, 22), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200))
                    if curActiveWindow.shape[0] > 0:
                        cv2.putText(textIm, "sound", (0, 33), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))                   
                    else:
                        cv2.putText(textIm, "silence", (0, 33), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,220))                   
                    cv2.imshow("Status", textIm)
                    cv2.moveWindow("Status", 50, 0)
                    midTermBuffer = []
                    ch = cv2.waitKey(10)
                    count += 1
                        

if __name__ == "__main__":
    args = parse_arguments()
    if args.task == "recordAndAnalyze":
        recordAudioSegments(args.blocksize, args.spectrogram, args.chromagram, args.recordactivity)        
