import tempfile
from io import BytesIO

import audiosegment
import matplotlib.pyplot as plt
import numpy as np
import pydub
import requests
import scipy.signal as sg
import sounddevice as sd


def speak(data):
    # We convert the mp3 bytes to wav.
    audio = pydub.AudioSegment.from_mp3(BytesIO(data))
    with tempfile.TemporaryFile() as fn:
        wavef = audio.export(fn, format='wav')
        wavef.seek(0)
        wave = wavef.read()
    # We get the raw data by removing the 24 first
    # bytes of the header.
    x = np.frombuffer(wave, np.int16)[24:] / 2. ** 15
    return x, audio.frame_rate


def play(data, samplerate, autoplay=False):
    sd.play(data, samplerate)
    sd.wait()


def highpass(cut_off_low, cut_off_high, x, fr):
    t = np.linspace(0., len(x) / fr, len(x))


    b, a = sg.butter(4, cut_off_low / (fr / 2.), 'low')
    x_fil_2 = sg.filtfilt(b, a, x)
    x_fil_2 = sg.sosfiltfilt(b, x)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(t, x, lw=1)

    ax.plot(t, x_fil_2, lw=1)
    play(x_fil_2, fr, autoplay=True)

    b, a = sg.butter(4, cut_off_high / (fr / 2.), 'high')
    x_fil = sg.filtfilt(b, a, x_fil_2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(t, x, lw=1)
    ax.plot(t, x_fil_2, lw=1)
    ax.plot(t, x_fil, lw=1)

    play(x_fil, fr, autoplay=True)

    print("2 filters")


if __name__ == '__main__':
    url = ('https://github.com/ipython-books/'
           'cookbook-2nd-data/blob/master/'
           'voice.mp3?raw=true')
    # voice = requests.get(url).content
    audio = audiosegment.from_file("data/test/nadi_ganga_hum.m4a")
    data = np.frombuffer(audio.raw_data, np.int16)
    frame_rate = audio.frame_rate

    # data, frame_rate = speak(voice)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    t = np.linspace(0., len(data) / frame_rate, len(data))
    ax.plot(t, data, lw=1)
    # play(data, frame_rate)
    #
    # b, a = sg.butter(4, 500. / (frame_rate / 2.), 'low')
    # x_fil = sg.filtfilt(b, a, data)
    # play(x_fil, frame_rate)
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.plot(t, data, lw=1)
    # ax.plot(t, x_fil, lw=1)
    #
    # b, a = sg.butter(4, 1000. / (frame_rate / 2.), 'high')
    # x_fil = sg.filtfilt(b, a, data)
    # play(x_fil, frame_rate)
    # fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    # ax.plot(t, data, lw=1)
    # ax.plot(t, x_fil, lw=1)

    highpass(600, 400, data, frame_rate)
    print("stop")
