import numpy as np

s = b'ASDAS VBCERT OYOGHJ QWERF GHJGH KTWSG '
dt = np.dtype(int)
dt = dt.newbyteorder('>')
data = np.frombuffer(s, dtype=np.uint8)
print(data)

asarray = np.asarray(data)
print(asarray.max())
# print(data - data.mean())
#
x = range(100)
#
# print(x)
# print([k for k in x[::2]])
# print([k for k in x[::3]])
# print([k for k in x[8:64:4]])

channels = []
for y in range(4):
    channels.append(data[y::1])
print(channels)

k = 7
k2 = (k -1) // 2
print(k2)
y = np.zeros ((len (asarray), k), dtype=asarray.dtype)

print(y)

assert asarray.ndim == 1

y[:,k2] = asarray
print(y)

for i in range(k2):
    j = k2 - i
    y[j:, i] = asarray[:-j]
    y[:j, i] = asarray[0]
    y[:-j, -(i + 1)] = asarray[j:]
    y[-j:, -(i + 1)] = asarray[-1]
print(np.median(y, axis=1))



log_ = np.log2(np.asarray([-1.00, 2 , 4]))
print(log_)