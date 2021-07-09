import audiosegment
import matplotlib.pyplot as plt
from dtw import *
from statsmodels.tsa.stattools import acf
import scipy.signal as sg

from audio_util import *


def get_pitches_from_audio(filename, multiplier=8):
    channels, frame_rate = get_channel_info_from_audio_file(filename)
    window_size = int(round(multiplier * frame_rate / 1000.0))
    data = channels[0]
    energy = get_energy(data)
    threshold = 0.15 * energy
    pitch_frequencies = []
    for _window in sliding_window(data, window_size, shift_ratio=0.8):
        pitch = get_frame_to_pitch(_window, frame_rate, threshold)
        pitch_frequencies.append(pitch)
    # print(pitch_frequencies)
    return pitch_frequencies


def get_channel_info_from_audio_file(filename):
    audio = audiosegment.from_file(filename)
    # audio = audio.resample(sample_rate_Hz=20000, sample_width=2)


    data = np.frombuffer(audio.raw_data, np.int16)

    fr = audio.frame_rate
    b, a = sg.butter(4, 1000. / (fr / 2.), 'high')
    data = sg.filtfilt(b, a, data)
    # b, a = sg.butter(4, 100. / (fr / 2.), 'low')
    # data = sg.filtfilt(b, a, data)

    data = data - data.mean()
    channels = []
    for i in range(audio.channels):
        channels.append(data[i::audio.channels])
    return channels, audio.frame_rate


def visualize(spect, frequencies, title=""):
    # Visualize the result of calling seg.filter_bank() for any number of filters
    i = 0
    for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
        plt.subplot(spect.shape[0], 1, index + 1)
    if i == 0:
        plt.title(title)
    i += 1
    plt.ylabel("{0:.0f}".format(freq))
    plt.plot(row)
    plt.show()
    seg = audiosegment.from_file("some_audio.wav").resample(sample_rate_Hz=24000, sample_width=2, channels=1)
    spec, frequencies = seg.filter_bank(nfilters=5)
    visualize(spec, frequencies)


def get_frame_to_pitch(frame, fs, threshold):
    frame_x = np.asarray(frame)
    invalid = -1
    if get_energy(frame_x) < threshold:
        return invalid
    else:
        down_limit = 40
        up_limit = 1000
        n1 = int(round(fs * 1.0 / up_limit))
        n2 = int(round(fs * 1.0 / down_limit))
        frame_acf = acf(frame_x, fft=True, nlags=1500)
        if n1 > frame_acf.size:
            n1 = frame_acf.size - 1
        frame_acf[np.arange(n1)] = invalid
        if n2 < frame_acf.size:
            frame_acf[np.arange(n2, frame_acf.size)] = invalid
        index = frame_acf.argmax()
        pitch = fs / index
        return pitch


def get_frame_to_pitch2(frame, fs, threshold):
    frame_x = np.asarray(frame)

    scipy.signal.stft(frame_x)

    invalid = -1
    if get_energy(frame_x) < threshold:
        return invalid
    else:
        # down_limit = 40
        # up_limit = 1000
        # n1 = int(round(fs * 1.0 / up_limit))
        # n2 = int(round(fs * 1.0 / down_limit))
        frame_acf = acf(frame_x, fft=True, nlags=1500)
        # if n1 > frame_acf.size:
        #     n1 = frame_acf.size - 1
        # frame_acf[np.arange(n1)] = invalid
        # if n2 < frame_acf.size:
        #     frame_acf[np.arange(n2, frame_acf.size)] = invalid
        # index = frame_acf.argmax()

        inflection = np.diff(np.sign(np.diff(frame_acf)))  # Find the second-order differences
        peaks = (inflection < 0).nonzero()[0] + 1  # Find where they are negative
        delay = peaks[frame_acf[peaks].argmax()]  # Of those, find the index with the maximum value
        pitch = fs / delay
        return pitch


def get_energy(sequence):
    data = np.asarray(sequence)
    return data.max()


def calculate_dtw(_model_pv, _query_pv):
    model_check = ~np.isnan(_model_pv)
    refined_model = _model_pv[model_check]

    query_check = ~np.isnan(_query_pv)
    refined_query = _query_pv[query_check]

    alignment = dtw(refined_query, refined_model, keep_internals=True)
    distance = alignment.distance
    # alignment.plot(type="threeway")
    return distance


def get_note_vector_by_file(file):
    frequencies = get_pitches_from_audio(file)
    notes = get_notes_by_frequencies(frequencies)
    return notes


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
