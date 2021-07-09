from dtw import *

alignment1 = dtw(["1", "3"], ["2", "2"], keep_internals=True)
print(alignment1.distance) # 2

alignment2 = dtw(["2", "3"], ["2", "3"], keep_internals=True)
print(alignment1.distance) # 0

# alignment3 = dtw(["2,4"], ["2", "3"], keep_internals=True)
# print(alignment3.distance) # could not convert string to float: '2,4'