# import librosa.output
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from scipy.fftpack import fft


def play(data, samplerate, autoplay=False):
    sd.play(data, samplerate)
    sd.wait()


def plot_wave(song, fs, bit_size=np.int8, same_plot=False):
    x = np.frombuffer(song.raw_data, bit_size)
    time_diff = np.linspace(0., len(x) / fs, len(x))
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(time_diff, x, lw=1)


def plot_freq(song, fs, bit_size=np.int16, same_plot=False):
    x = np.frombuffer(song.raw_data, bit_size)
    time_diff = np.linspace(0., len(x) / fs, len(x))

    fft1 = fft(x)
    fft_abs_half = np.abs(fft1[0: int(len(x) / 2)])

    xf = np.linspace(0.0, len(x) / fs, len(x) // 2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(xf, fft_abs_half, lw=1)
    print("Freq")


def plot_in_same(song, axes, pos, alpha=0.5, bit_size=np.int8):
    x = np.frombuffer(song.raw_data, bit_size)
    time = np.linspace(0., len(x) / fr, len(x))
    axes[pos].plot(time, x, lw=1, alpha=alpha)
    print("plot")


def signal_wave_approach1():
    # read in audio file and get the two mono tracks
    sound_stereo = AudioSegment.from_file('data/sinhala/Mal Mitak Thiyanna.wav', format="wav")
    fr = sound_stereo.frame_rate

    plot_freq(sound_stereo, fr)

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    plot_in_same(sound_stereo, ax, 0, bit_size=np.int16)

    sound_monoL = sound_stereo.split_to_mono()[0]
    plot_in_same(sound_monoL, ax, 1, bit_size=np.int8)
    sound_monoR = sound_stereo.split_to_mono()[1]
    plot_in_same(sound_monoR, ax, 1, bit_size=np.int8)

    # play(np.frombuffer(sound_monoL.raw_data, np.int8), fr, autoplay=True)
    # play(np.frombuffer(sound_monoR.raw_data, np.int8), fr, autoplay=True)
    # Invert phase of the Right audio file
    sound_monoR_inv = sound_monoR.invert_phase()
    plot_in_same(sound_monoR_inv, ax, 1, bit_size=np.int8)
    # play(np.frombuffer(sound_monoR_inv.raw_data, np.int8), fr, autoplay=True)

    # Merge two L and R_inv files, this cancels out the centers
    sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

    # Export merged audio file
    # fh = sound_CentersOut.export(myAudioFile_CentersOut, format="mp3")
    data = np.frombuffer(sound_CentersOut.raw_data, np.int8)

    plot_in_same(data, ax, 1, bit_size=np.int8)

    # play(data, fr, autoplay=True)

    # new_y = librosa.istft(S_foreground * phase)
    # librosa.output.write_wav("./new-audio.wav", new_y, sr)



if __name__ == '__main__':
    sound_stereo = AudioSegment.from_file('data/sinhala/Mal Mitak Thiyanna.wav', format="wav")
    fr = sound_stereo.frame_rate
    data = np.frombuffer(sound_stereo.raw_data, np.int16)



