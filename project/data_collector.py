# This file will be used to record and tag audio samples for our knock detection model


import pyaudio
import wave
import pandas as pd
import os


# Audio recording parameters ----------------------------------------------------#
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 20000
record_seconds = 8
# -------------------------------------------------------------------------------#

def record_audio(filename, chunk, sample_rate, record_seconds):
    # This is a straight copy of https://www.thepythoncode.com/article/play-and-record-audio-sound-in-python

    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()


def random_string():
    # This function will generate a random string of 10 characters
    # This will be used to name our audio files
    import random
    import string
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(10))
    return result_str

# make a dataframe of filenames and their labels
dfFilename = "data/knockdata.pkl"
dfFilenameCSV = "data/knockdata.csv"
df = None
if(os.path.isfile(dfFilename)):
    df = pd.read_pickle(dfFilename)
else:
    df = pd.DataFrame(columns=["filename", "label"])


folder = "data/audio"

while True:
    # tag should be a random string of 10 characters
    tag = random_string()
    # Ask the user to press enter to record
    input("Press enter to record...")
    # Record the audio
    record_audio(f"{folder}/{tag}.wav", chunk, sample_rate, record_seconds)

    # Press d to delete the recording
    # Press any other key to save the recording
    i = input("Press d to delete, any other key to save: ")
    if(i == "d"):
        # Delete the recording
        os.remove(f"{folder}/{tag}.wav")
    else:
        # Ask the user to enter the label
        label = input("Enter the label: ")
        # Add the filename and label to the dataframe
        df = pd.concat([df, pd.DataFrame([[f"{tag}.wav", label]], columns=["filename", "label"])])
        # Save the dataframe
        df.to_pickle(dfFilename)
    
    # Ask the user if they want to record another sample
    if input("Record another sample? (y/n): ") == "n":
        break

# Save the dataframe
df.to_csv(dfFilenameCSV)
