# This file will be used to record and tag audio samples for our knock detection model


import pyaudio
import wave
import pandas as pd
import os
import collections
import threading
import time
import numpy as np

# Load checkpoint
from tensorflow.keras.models import load_model
#Import the Model from keras
from tensorflow.keras.models import Model

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
            self.bufferCopy = self.buffer.copy()
            fm = frameCntr
            bufferLock.release()

            last = self.bufferCopy[-1]
            self.avgs.append(np.mean(np.frombuffer(last, dtype=np.int16)))
            
            avg = np.mean(self.avgs)
            avg = avg if avg > 200 else 200

            for val in np.frombuffer(last, dtype=np.int16):
                if val > avg * 5:
                    # print("Knock detected, current frame : {} and match set at {}".format(fm, fm + 160))
                    self.callMatchList.append(fm + 160)
                    break
            
            if(fm in self.callMatchList):
                matchQueueLock.acquire()
                # Get last n elements from bufferCopy
                l = [self.bufferCopy[i] for i in range(len(self.bufferCopy) - 162, len(self.bufferCopy))]
                # print("Here fm = {} and len(l) = {}".format(fm, len(l)))
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

        self.compressionModelFileName = "modelCheckpoints/Weights-smallmodel2-315--0.00202.hdf5"
        self.classificationModelFileName = "modelCheckpoints/classificationModel.h5"

        # Load the models using tensorflow
        self.autoencoder = load_model("modelCheckpoints/Weights-smallmodel2-315--0.00202.hdf5")
        self.encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('dense_3').output)
        self.classificationModel = load_model("modelCheckpoints/classificationModel.h5")

        self.passwordFile = "data/compressedSignalsAdjusted/aejvqoknnp.npy"
        self.password = np.load(self.passwordFile)

    def run(self):
        print("Starting " + self.name)
        while(True):
            matchWorkQueueSem.acquire()
            matchQueueLock.acquire()
            self.bufferCopy = matchWorkQueue.popleft()
            matchQueueLock.release()

            # print("Matching knock")

            # Flatten the bufferCopy
            flattenedBuffer = np.array([np.frombuffer(x, dtype=np.int16) for x in self.bufferCopy])
            flattenedBuffer = flattenedBuffer.flatten()
            flattenedBuffer = flattenedBuffer[: 20000 * 8]

            # Compress
            compressed = self.compress_signal(flattenedBuffer)

            # predict X
            X = np.array([np.concatenate((self.password, compressed), axis=0)])


            # Predict
            y = self.classificationModel.predict(X, verbose=0)

            if(y[0] > 0.8):
                print("THERE WAS A MATCH WITH A SCORE OF {}\n\n".format(y[0]))



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

    

    def stop(self):
        print("Stopping " + self.name)
        self._stop_event.set()

# Create new threads
thread1 = RecordAudio(1, "Audio recording", 1, buffer, chunk, sample_rate)
thread2 = CheckForKnock(2, "Knock detection", 2, buffer)
thread3 = MatchKnock(3, "ML Thread", 3, buffer)

# Start new Threads
# Pass in the buffer to the thread
thread1.start()
thread2.start()
thread3.start()

useInp = input("Press enter to stop")
if(useInp == ""):
    thread1.stop()
    thread2.stop()
    thread3.stop()

    # Wait for threads to finish
    thread1.join()
    thread2.join()
    thread3.join()

print("Exiting Main Thread")

