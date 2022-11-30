# This file will be used to record and tag audio samples for our knock detection model


import pyaudio
import wave
import pandas as pd
import os
import collections
import threading
import time
import numpy as np
import copy

# Load checkpoint
from tensorflow.keras.models import load_model
#Import the Model from keras
from tensorflow.keras.models import Model
import librosa
import matplotlib.pyplot as plt
# funcanimation
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Audio recording parameters ----------------------------------------------------#
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
sample_rate = 20000
buffer = collections.deque(maxlen=(180))
frameCntr = 0
matchWorkQueue = collections.deque(maxlen=10)
# -------------------------------------------------------------------------------#


# Make three threads
# 1. Record audio
# 2. Check for knock
# 3. Use tensorflow to match knock


# To notify the knock detection thread that a new frame of audio has been recorded
newFrameLock = threading.Lock()

# To access the common audio recording buffer
bufferLock = threading.Lock()

# To write to work queue for matching thread
matchQueueLock = threading.Lock()

# To notify the matching thread
matchWorkQueueSem = threading.Semaphore(0)

# Match found semaphore used by matching thread to notify plotting thread
matchFoundSemLock = threading.Lock()
matchFound = 0
class RecordAudio(threading.Thread):
    def __init__(self, threadID, name, counter,  buffer, chunk, sample_rate):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.buffer = buffer
        self.chunk = chunk
        self.sample_rate = sample_rate
        self._stop_event = threading.Event()

    def run(self):
        global frameCntr
        print("Starting " + self.name)

        # initialize PyAudio object
        self.p = pyaudio.PyAudio()
        # open stream object as input & output
        self.stream = self.p.open(format=FORMAT,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)
        
        while(True):
            data = self.stream.read(self.chunk)
            bufferLock.acquire()
            self.buffer.append(data)
            frameCntr += 1
            bufferLock.release()

            # Notify the knock detection thread that a new frame is available
            if(newFrameLock.locked()):
                newFrameLock.release()


            # If parent wants us dead
            if(self._stop_event.is_set()):
                return

    def stop(self):
        print("Stopping " + self.name)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        self._stop_event.set()

class CheckForKnock(threading.Thread):
    def __init__(self, threadID, name, counter, buffer):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.avgs = collections.deque(maxlen=10)
        self.buffer = buffer
        self.bufferCopy = None
        self._stop_event = threading.Event()
        
        self.callMatchList = collections.deque(maxlen=10)

    def run(self):
        print("Starting " + self.name)
        while(True):
            if(len(self.buffer) < 1):
                continue

            bufferLock.acquire()
            self.bufferCopy = [copy.copy(x) for x in self.buffer]
            fm = frameCntr
            bufferLock.release()

            last = self.bufferCopy[-1]
            self.avgs.append(np.mean(np.frombuffer(last, dtype=np.int16)))
            
            avg = np.mean(self.avgs)
            avg = avg if avg > 500 else 500

            for val in np.frombuffer(last, dtype=np.int16):
                if val > avg * 3:
                    # print("Knock detected, current frame : {} and match set at {}".format(fm, fm + 160))
                    self.callMatchList.append(fm + 160)
                    break
            
            if(fm in self.callMatchList):
                matchQueueLock.acquire()
                # Get last n elements from bufferCopy
                l = [copy.copy(self.bufferCopy[i]) for i in range(len(self.bufferCopy) - 162, len(self.bufferCopy))]

                matchWorkQueue.append(l)
                matchQueueLock.release()
                matchWorkQueueSem.release()

                # remove fm from callMatchList
                self.callMatchList.remove(fm)

            if(self._stop_event.is_set()):
                return

            newFrameLock.acquire()


    def stop(self):
        print("Stopping " + self.name)
        self._stop_event.set()


class MatchKnock(threading.Thread):
    def __init__(self, threadID, name, counter, buffer):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.buffer = buffer
        self.bufferCopy = None
        self._stop_event = threading.Event()

        # self.compressionModelFileName = "modelCheckpoints/Weights-smallmodel2-315--0.00202.hdf5"
        # self.classificationModelFileName = "modelCheckpoints/classificationModel.h5"

        # # Load the models using tensorflow
        # self.autoencoder = load_model("modelCheckpoints/Weights-smallmodel2-315--0.00202.hdf5")
        # self.encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('dense_3').output)
        # self.classificationModel = load_model("modelCheckpoints/classificationModel.h5")

        # self.passwordFile = "data/compressedSignalsAdjusted/cgxwaehnir.npy"
        # self.password = np.load(self.passwordFile)

        self.featureExtrationModelFile = "modelCheckpoints/featureExtractionModel_2.h5"
        self.featureExtractionModel = load_model(self.featureExtrationModelFile)

        # self.passwordFileForFeatureExtraction = 'data/features/qdcdgqjjdr.npy'
        self.passwordFiles = [
            'data/features/qdcdgqjjdr.npy',
            'data/features/auekkfhgvs.npy',
            'data/features/haflvkblwb.npy'
        ]


        self.passwords = [np.load(x) for x in self.passwordFiles]

        # self.passwordForFeatureExtraction = np.load(self.passwordFileForFeatureExtraction)
        # self.passwordForFeatureExtraction = np.pad(self.passwordForFeatureExtraction, (0, 25 - len(self.passwordForFeatureExtraction)), 'constant')

        self.passwords = [np.pad(x, (0, 25 - len(x)), 'constant') for x in self.passwords]

    def run(self):
        global matchFound
        print("Starting " + self.name)
        while(True):
            matchWorkQueueSem.acquire()
            matchQueueLock.acquire()
            self.bufferCopy = matchWorkQueue.popleft()
            matchQueueLock.release()

            # Flatten the bufferCopy
            flattenedBuffer = np.array([np.frombuffer(x, dtype=np.int16) for x in self.bufferCopy])
            flattenedBuffer = flattenedBuffer.flatten()
            flattenedBuffer = flattenedBuffer[: 20000 * 8]

            # Compress
            # compressed = self.compress_signal(flattenedBuffer)
            # # predict X
            # X = np.array([np.concatenate((self.password, compressed), axis=0)])
            # # Predict
            # y = self.classificationModel.predict(X, verbose=0)


            # if(y[0] > 0.8):
            #     print("THERE WAS A MATCH WITH A SCORE OF {}\n\n".format(y[0]))


            featureBuffer = flattenedBuffer.astype(np.float32)
            # write fearure buffer to file
            self.save_audio(featureBuffer, "data/temp.wav")
            normalizedBuffer = librosa.load("data/temp.wav", sr=20000)[0]
            # normalizedBuffer = featureBuffer / np.max(np.abs(featureBuffer))
            features = self.get_features(normalizedBuffer)
    
            # XForFeatureExtraction = np.array([np.concatenate((self.passwordForFeatureExtraction, features), axis=0)])

            XForFeatureExtraction = []
            for password in self.passwords:
                XForFeatureExtraction.append(np.concatenate((password, features), axis=0))


            XForFeatureExtraction = np.array(XForFeatureExtraction)

            y2 = self.featureExtractionModel.predict(XForFeatureExtraction, verbose=0)

            print("Passwords are : {}".format(self.passwords))
            print("Features are : {}".format(features))
            print(y2)

            if(np.max(y2) > 0.8):
                print("THERE WAS A MATCH WITH A SCORE OF {}\n\n".format(np.max(y2)))
                print(y2)
                                # set matchfoundsem to 3
                matchFoundSemLock.acquire()
                if(matchFound == 0):
                    matchFound = 20
                matchFoundSemLock.release()

            # if(y2[0] > 0.8):
            #     print("THERE WAS A MATCH IN FEATURE MODEL WITH A SCORE OF {}\n\n".format(y2[0]))
            #     # set matchfoundsem to 3
            #     matchFoundSemLock.acquire()
            #     if(matchFound == 0):
            #         matchFound = 20
            #     matchFoundSemLock.release()



            if(self._stop_event.is_set()):
                return

    def split_into_peices(self, signal, size):
        pieces = []

        i = 0
        while i < len(signal):
            if i + size < len(signal):
                pieces.append(np.array(signal[i:i+size]))
            else:
                t = np.array(signal[i:])
                t = np.pad(t, (0, size - len(t)), 'constant')
                pieces.append(t)

            i += size

        return pieces

    def compress_signal(self, signal):
        # split normalised signals into 1000 long segments for prediction
        pieces = self.split_into_peices(signal, 1000)

        encoded = self.encoder.predict(np.array(pieces), verbose=0)
        
        return np.concatenate(encoded, axis=0)

    
    def get_features(self, x):
        poly_features=librosa.feature.poly_features(y = x) #order 1 by default
        features = poly_features[1]
        # if val inn norm is > 2, set to 1, else  0
        # Everything over 2000 hertz gets set to 1
        features_array = features > 2
        features_array = features_array.astype(int)

        # find changes from 0 to 1
        features_array = np.diff(features_array)
        # find indices where changes occur
        features_array = np.where(features_array == 1)[0]

        difference_array = np.diff(features_array)
        
        # Pad to length of 15
        try:
            difference_array = np.pad(difference_array, (0, 25 - len(difference_array)), 'constant')
        except:
            difference_array = [0] * 25

        return difference_array

    def stop(self):
        print("Stopping " + self.name)
        self._stop_event.set()

    def save_audio(self, signal, filename, denormalize=False):
        if denormalize:
            signal = signal * 30000
        
        signal = np.array(signal, dtype=np.int16)
        wavefile = wave.open(filename, "w")
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(20000)
        wavefile.writeframes(signal)
        wavefile.close()


class Plotter(threading.Thread):
    def __init__(self, threadID, name, counter, buffer):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.bufferCopy = None
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.buffer = buffer
    def animate(self, i):
        global matchFound
        # Get the data from the buffer
        # Plot the data
        # Return the line
        bufferLock.acquire()
        self.bufferCopy = [copy.copy(i) for i in self.buffer]
        bufferLock.release()

        # Flatten the bufferCopy
        flattenedBuffer = np.array([np.frombuffer(x, dtype=np.int16) for x in self.bufferCopy])
        flattenedBuffer = flattenedBuffer.flatten()

        self.line.set_data(np.arange(len(flattenedBuffer)), flattenedBuffer)


        matchFoundSemLock.acquire()
        mf = matchFound
        if(matchFound > 0):
            matchFound -= 1
        matchFoundSemLock.release()

        if(mf > 0):
            self.line.set_color('green')
            self.line.set_linewidth(5)
        else:
            self.line.set_color('red')
            self.line.set_linewidth(2)

        return self.line,

    def initPlot(self):
        self.line, = self.ax1.plot([], [], lw=2, c='red')
        return self.line,

    def run(self):
        print("Starting " + self.name)
        # Make funanimation
        fig = plt.figure()
        self.ax1 = plt.axes(xlim=(0, 180 * 1024), ylim=(-32768, 32768))
        self.ani = animation.FuncAnimation(fig, self.animate, init_func=self.initPlot, interval=50, blit=True)
        plt.show()

    def stop(self):
        print("Stopping " + self.name)
        self._stop_event.set()
        self.ani.event_source.stop()


# Create new threads
thread1 = RecordAudio(1, "Audio recording", 1, buffer, chunk, sample_rate)
thread2 = CheckForKnock(2, "Knock detection", 2, buffer)
thread3 = MatchKnock(3, "ML Thread", 3, buffer)
thread4 = Plotter(4, "Plotter", 4, buffer)
# Start new Threads
# Pass in the buffer to the thread
thread1.start()
thread2.start()
thread3.start()
thread4.start()



while(thread4.is_alive()):
    time.sleep(1)


thread1.stop()
thread2.stop()
thread3.stop()


# Wait for threads to finish
thread1.join()
thread2.join()
thread3.join()
thread4.join()

print("Exiting Main Thread")

