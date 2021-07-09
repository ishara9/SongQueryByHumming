import audiosegment
import numpy as np
from scipy import fftpack
from statsmodels.tsa.stattools import acf
from scipy.signal import medfilt
import librosa
import librosa.display
import matplotlib.pyplot as plt

# list1 = [1, 2, 3, 4, 5, 6, 7]
# list1[0::2]

def gen_seq(frame_count, shift, window_size):
    for i in range(0, int(frame_count) * shift, shift):
        yield sequence[i:i + window_size]


def get_frame_to_pitch(_window, frame_rate, threshold):
    frame_x = np.asarray(_window)
    if frame_x.max() < threshold:
        return -1
    else:
        frame_acf = acf(frame_x, fft=True, nlags=40)


def get_energy(seq):
    energy = np.asarray(seq).max()
    return energy


def get_frame_to_pitch(frame, fs, threshold):
    frame_x = np.asarray(frame)
    invalid = -1
    frame_energy = get_energy(frame_x)
    if frame_energy < threshold:
        return invalid
    else:
        down_limit = 40
        up_limit = 1000
        n1 = int(round(fs * 1.0 / up_limit))
        n2 = int(round(fs * 1.0 / down_limit))
        frame_acf = acf(frame_x, fft=True, nlags=1500)
        # if n1 > frame_acf.size:
        #     n1 = frame_acf.size - 1
        # frame_acf[np.arange(n1)] = invalid
        # if n2 < frame_acf.size:
        #     frame_acf[np.arange(n2, frame_acf.size)] = invalid
        index = frame_acf.argmax()

        inflection = np.diff(np.sign(np.diff(frame_acf)))  # Find the second-order differences
        peaks = (inflection < 0).nonzero()[0] + 1  # Find where they are negative
        delay = peaks[frame_acf[peaks].argmax()]  # Of those, find the index with the maximum value

        pitch = fs / delay
        return pitch


def get_frame_to_pitch2(frame, fs, threshold):
    frame_x = np.asarray(frame)
    fftpack.fft(frame_x)
    X = librosa.stft(frame_x)
    Xdb = librosa.amplitude_to_db(abs(X))  # convert an amplitude spectrogram to dB-scaled spectrogram.

    fs = 16384

    n_fft = 256

    np.arange(0, 1 + n_fft / 2) * fs / n_fft
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)

    return pitch


def get_notes_by_frequencies(frequencies):
    # log base: 2
    log440 = 8.78135971
    notes_array = np.asarray(frequencies)
    np.seterr(all='ignore')
    try:
        notes = 12 * (np.log2(notes_array) - log440)
        return notes
    except Exception as e:
        raise Exception("freq to note conversion error: " + str(e))


def filter_outlier_pitches(pv, threshold=10):
    adjusted = pv - np.median(pv)
    loc = (abs(adjusted) > threshold)
    pv[loc] = 0
    return pv


def get_channel_info_from_audio_file_lib(filename):
    audio = audiosegment.from_file(filename)

    hist_bins, hist_values = audio.fft()
    hist_vals_real_normed = np.abs(hist_values) / len(hist_values)
    # plt.plot(hist_bins / 1000, hist_vals_real_normed)
    # plt.xlabel("kHz")
    # plt.ylabel("dB")
    # plt.show()

    data = hist_vals_real_normed
    channels2 = []
    for i in range(audio.channels):
        channels2.append(data[i::audio.channels])

    # freqs, times, amplitudes =  audio.spectrogram(window_length_s=0.03, overlap=0.5)
    #
    # amplitudes = 10 * np.log10(amplitudes + 1e-9)
    # # Plot
    # plt.pcolormesh(times, freqs, amplitudes)
    # plt.xlabel("Time in Seconds")
    # plt.ylabel("Frequency in Hz")
    # plt.show()

    # data = np.frombuffer(audio.raw_data, np.int16)
    # data = data - data.mean()
    # channels = []
    # for i in range(audio.channels):
    #     channels.append(data[i::audio.channels])
    return channels2, audio.frame_rate


audio = audiosegment.from_file("data/train/A.wav")
# audio = audio.resample(sample_rate_Hz=20000, sample_width=2)
data = np.frombuffer(audio.raw_data, np.int16)

data = data - data.mean()
channels = []
for i in range(audio.channels):
    channels.append(data[i::audio.channels])

frame_rate = audio.frame_rate
multiplier = 8
window_size = int(round(multiplier * frame_rate / 40))
shift_ratio = 1
shift = int(shift_ratio * window_size)
sequence = channels[0]
energy = np.asarray(sequence).max()
threshold = energy * 0.1
frame_count = ((len(sequence) - window_size) / shift) + 1
pitch_frequencies = []
for _window in gen_seq(frame_count, shift, window_size):
    pitch = get_frame_to_pitch(_window, frame_rate, threshold)
    pitch_frequencies.append(pitch)
print(pitch_frequencies)
notes = get_notes_by_frequencies(pitch_frequencies)
print(notes)
notes2 = filter_outlier_pitches(notes)
notes_diff = np.diff(notes2)
