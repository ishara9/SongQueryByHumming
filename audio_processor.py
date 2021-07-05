import audiosegment
import matplotlib.pyplot as plt
# from pydub import AudioSegment
from dtw import *
from scipy.signal import medfilt
from statsmodels.tsa.stattools import acf

from audio_util import *


def get_pitches_from_audio(filename, multiplier=16):
    channels, frame_rate = get_channel_info_from_audio_file(filename)
    window_size = int(round(multiplier * frame_rate / 1000.0))
    data = channels[0]
    energy = get_energy(data)
    threshold = 0.3 * energy
    pitch_frequencies = []
    for _window in sliding_window(data, window_size):
        pitch = get_frame_to_pitch(_window, frame_rate, threshold)
        pitch_frequencies.append(pitch)
    return pitch_frequencies


def get_pitches_from_audio2(filename, multiplier=40):
    # channels, frame_rate = get_channel_info_from_audio_file(filename)
    # window_size = int(round(multiplier * frame_rate / 1000.0))
    # data = channels
    # energy = get_energy(data)
    # threshold = 0.3 * energy
    # pitch_frequencies = []
    # for _window in sliding_window(data, window_size):
    #     pitch = get_frame_to_pitch(_window, frame_rate, threshold)
    #     pitch_frequencies.append(pitch)

    import librosa.display
    song, sr = librosa.load(filename)
    n_fft = 2048
    pitch_frequencies = np.abs(librosa.stft(song))
    return pitch_frequencies


def get_channel_info_from_audio_file(filename):
    audio = audiosegment.from_file(filename)
    audio = audio.resample(sample_rate_Hz=20000, sample_width=2)
    data = np.frombuffer(audio.raw_data, np.int16)
    data = data - data.mean()
    channels = []
    for i in range(audio.channels):
        channels.append(data[i::audio.channels])
    return channels, audio.frame_rate


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
        frame_acf = acf(frame_x, fft=True, nlags=40)
        if n1 > frame_acf.size:
            n1 = frame_acf.size - 1
        frame_acf[np.arange(n1)] = invalid
        if n2 < frame_acf.size:
            frame_acf[np.arange(n2, frame_acf.size)] = invalid
        index = frame_acf.argmax()
        pitch = fs / (index - 1.0)
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


def get_pitch_vector_by_file(file):
    frequencies = get_pitches_from_audio(file)
    notes = get_notes_by_frequencies(frequencies)
    pitch_vector = medfilt(notes)
    return pitch_vector


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


if __name__ == '__main__':
    file = get_channel_info_from_audio_file_lib("data/test/nadi_ganga_hum.m4a")
