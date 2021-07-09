import audiosegment
import numpy as np
from dtw import dtw

# s = b'ASDAS VBCERT OYOGHJ QWERF GHJGH KTWSG '
# dt = np.dtype(int)
# dt = dt.newbyteorder('>')
# data = np.frombuffer(s, dtype=np.uint8)
# print(data)
#
# asarray = np.asarray(data)
# print(asarray.max())
# # print(data - data.mean())
# #
# x = range(100)
# #
# # print(x)
# # print([k for k in x[::2]])
# # print([k for k in x[::3]])
# # print([k for k in x[8:64:4]])
#
# channels = []
# for y in range(4):
#     channels.append(data[y::1])
# print(channels)
#
# k = 7
# k2 = (k -1) // 2
# print(k2)
# y = np.zeros ((len (asarray), k), dtype=asarray.dtype)
#
# print(y)
#
# assert asarray.ndim == 1
#
# y[:,k2] = asarray
# print(y)
#
# for i in range(k2):
#     j = k2 - i
#     y[j:, i] = asarray[:-j]
#     y[:j, i] = asarray[0]
#     y[:-j, -(i + 1)] = asarray[j:]
#     y[-j:, -(i + 1)] = asarray[-1]
# print(np.median(y, axis=1))

# log_ = np.log2(np.asarray([-1.00, 2 , 4]))
# print(log_)

import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.io import wavfile
from skimage import util


# plt.style.use('style/elegant.mplstyle')

f = 10  # Frequency, in cycles per second, or Hertz
f_s = 100  # Sampling rate, or number of measurements per second

t = np.linspace(0, 2, 2 * f_s, endpoint=False)
x = np.sin(f * 2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, x)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal amplitude')

X = fftpack.fft(x)
freqs = fftpack.fftfreq(len(x)) * f_s

fig, ax = plt.subplots()

ax.stem(freqs, np.abs(X))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-f_s / 2, f_s / 2)
ax.set_ylim(-5, 110)


rate, audio = wavfile.read('data/english/This Is Me.wav')

N = audio.shape[0]
L = N / rate

print(f'Audio length: {L:.2f} seconds')

f, ax = plt.subplots()
ax.plot(np.arange(N) / rate, audio)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')


M = 1024

slices = util.view_as_windows(audio, window_shape=(M,), step=100)
print(f'Audio shape: {audio.shape}, Sliced audio shape: {slices.shape}')

audio = audiosegment.from_file("data/test/00020.wav")

#
#
# x = _query_pv[query_check]
# y = _query_pv[query_check]
#
# alignment = dtw(x, y, keep_internals=True)
# distance = alignment.distance
# # alignment.plot(type="threeway")