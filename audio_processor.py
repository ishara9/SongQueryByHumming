import audiosegment
import librosa
import matplotlib.pyplot as plt
from dtw import *
from statsmodels.tsa.stattools import acf
import scipy.signal as sg

from DTW_handler import local_dtw_distance
from audio_util import *
from logger import log_time


def get_pitches_from_audio(filename, multiplier=8):
    log_time("get_pitches_from_audio Start")
    channels, frame_rate, data = get_channel_info_from_audio_file(filename)
    # window_size = int(round(multiplier * frame_rate / 5000.0))
    # data = channels[0]
    # energy = get_energy(data)
    # threshold = 0.15 * energy
    # pitch_frequencies = []
    # for _window in sliding_window(data, window_size, shift_ratio=1):
    #     pitch = get_frame_to_pitch_acf(_window, frame_rate, threshold)
    #     pitch_frequencies.append(pitch)
    # print(pitch_frequencies)

    time_per_hop = 32 / 10000
    # window_size = int(round(multiplier * frame_rate / 5000))
    window_size = int(round(frame_rate * time_per_hop))
    shift_ratio = 1
    shift = int(shift_ratio * window_size)
    sequence = channels[0]
    energy = np.asarray(sequence).max()
    threshold = energy * 0.3
    hop_count = ((len(sequence) - window_size) / shift) + 1
    pitch_freq_hops = []
    log_time("get_pitches_from_audio:gen_seq Start")
    for _window in gen_seq(hop_count, shift, window_size, sequence):
        pitch = get_frame_to_pitch_acf(_window, frame_rate, threshold)
        pitch_freq_hops.append(pitch)
    log_time("get_pitches_from_audio:gen_seq End")
    # print(pitch_freq_hops)
    time_per_hop = len(sequence) / (frame_rate * hop_count)

    y = np.asarray(np.double(data))
    sr = frame_rate
    # y, sr = librosa.load('data/selected_set/Roo Sara C.wav')
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    pitch_freq_hops = avg_filter(pitch_freq_hops, time_per_hop, tempo[0])

    log_time("get_pitches_from_audio End")
    return pitch_freq_hops


def gen_seq(frame_count, shift, window_size, sequence):
    log_time("gen_seq:" + str(frame_count))
    for i in range(0, int(frame_count) * shift, shift):
        yield sequence[i:i + window_size]


def avg_filter(pitch_freq_hops, time_per_hop, tempo):
    hop_count = len(pitch_freq_hops)
    beats_per_second = tempo / 60.0  # beats per second
    hops_per_second = 1 / time_per_hop
    hops_per_beat_window = round((hops_per_second - beats_per_second) / beats_per_second + 1)
    divisor = 1
    hops_per_beat_window = round(hops_per_beat_window / divisor)
    iterations = int(hop_count / hops_per_beat_window) + 1
    for x in range(iterations):
        start = x * hops_per_beat_window
        end = start + hops_per_beat_window
        window_data = pitch_freq_hops[start:end]
        new_end = start + len(window_data)
        local_total=0
        for s in range(len(window_data)):
            local_total = local_total + window_data[s]
        average_per_window = local_total / (new_end - start)
        pitch_freq_hops[start:new_end] = [average_per_window] * (new_end - start)
    return pitch_freq_hops


def max_filter(pitch_freq_hops, time_per_hop, tempo):
    log_time("max_filter Start")
    hop_count = len(pitch_freq_hops)

    # beats_per_second = 2.267  # per second
    beats_per_second = tempo/60.0  # per second

    hops_per_second = 1 / time_per_hop  # 625

    hops_per_beat_window = round(hops_per_second / beats_per_second)

    iterations = round((hop_count - hops_per_beat_window) / hops_per_beat_window)  # 22

    log_time("loop:iterations Start")
    for x in range(iterations):
        start = x * hops_per_beat_window
        end = start + hops_per_beat_window
        window_data = pitch_freq_hops[start:end]
        # new_end = start + len(window_data)
        local_max = -100
        for s in range(len(window_data)):
            if window_data[s] > local_max:
                local_max = window_data[s]
            else:
                pitch_freq_hops[start + s] = local_max
        # pitch_freq_hops[start:new_end] = [local_max] * (new_end-start)
    log_time("loop:iterations End")
    log_time("max_filter End")
    return pitch_freq_hops


def get_channel_info_from_audio_file(filename):
    log_time("get_channel_info_from_audio_file Start")
    audio = audiosegment.from_file(filename)
    # audio = audio.resample(sample_rate_Hz=20000, sample_width=2)

    data = np.frombuffer(audio.raw_data, np.int16)

    fr = audio.frame_rate

    # log_time("Upper band cut Start")
    # b, a = sg.butter(4, 10000. / (fr / 2.), 'high')
    # data = sg.filtfilt(b, a, data)
    # log_time("Upper band cut End")

    log_time("Lower band cut Start")
    d, c = sg.butter(4, 800. / (fr / 2.), 'low')
    data = sg.filtfilt(d, c, data)
    log_time("Lower band cut End")

    data = data - data.mean()
    channels = []
    for i in range(audio.channels):
        channels.append(data[i::audio.channels])

    log_time("get_channel_info_from_audio_file End")
    return channels, audio.frame_rate, data


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


def get_frame_to_pitch_acf(frame, fs, threshold):
    frame_x = np.asarray(frame)
    invalid = -1
    frame_energy = get_energy(frame_x)
    if frame_energy < threshold:
        return invalid
    else:
        frame_acf = acf(frame_x, fft=True, nlags=1500)
        inflection = np.diff(np.sign(np.diff(frame_acf)))  # Find the second-order differences
        peaks = (inflection < 0).nonzero()[0] + 1  # Find where they are negative
        pitch1 = 0
        if len(frame_acf[peaks]) > 0:
            delay = peaks[frame_acf[peaks].argmax()]  # Of those, find the index with the maximum value
            pitch1 = fs / delay
        return pitch1


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
    log_time("calculate_dtw start")
    # model_check = ~np.isnan(_model_pv)
    # refined_model = _model_pv[model_check]

    # query_check = ~np.isnan(_query_pv)
    # refined_query = _query_pv[query_check]

    # distance = local_dtw_distance(refined_query, refined_model)
    alignment = dtw(_query_pv, _model_pv, keep_internals=True)
    distance = alignment.distance
    log_time("distanc: " + str(distance))
    # alignment.plot(type="threeway")
    log_time("calculate_dtw end")
    return distance


def get_note_vector_by_file(file):
    log_time("get_note_vector_by_file Start")
    frequencies = get_pitches_from_audio(file)
    # plot_freq_time(pitch_freq_hops, time_per_hop)

    notes = get_notes_by_frequencies(frequencies)
    # print(notes)
    notes = filter_outlier_pitches(notes)
    # plot_freq_time(notes, time_per_hop)
    notes = refine_model(notes)
    # notes = notes[notes != 0]
    log_time("get_note_vector_by_file End")
    return notes


def refine_model(model):
    log_time("refine_model Start")
    model_check = ~np.isnan(model)
    refined_model = model[model_check]
    model_check = ~np.isinf(refined_model)
    refined_model = refined_model[model_check]
    log_time("refine_model End")
    return refined_model


def get_notes_by_frequencies(frequencies):
    log_time("get_notes_by_frequencies Start")
    # log base: 2
    log440 = 8.78135971
    notes_array = np.asarray(frequencies)
    np.seterr(all='ignore')
    try:
        notes = 12 * (np.log2(notes_array) - log440)
        log_time("get_notes_by_frequencies End")
        return notes
    except Exception as e:
        log_time("get_notes_by_frequencies Error: " + str(e))
        raise Exception("freq to note conversion error: " + str(e))
