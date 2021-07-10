from dtw import *

from logger import log_time


def local_dtw_distance(query, source):
    min_dtw = float('inf')
    for distance in dtw_sliding_window(query, source):
        if distance < min_dtw:
            min_dtw = distance
    log_time("min dtw: " + str(min_dtw))
    return min_dtw


def dtw_sliding_window(query, source, hop_size=50):
    hop_size = len(query)-1
    iterations = ((len(source) - len(query)) / hop_size) + 1
    if iterations < 0:
        iterations = 1
    for x in range(0, int(iterations) * hop_size, hop_size):
        distance = dtw(query, source[x:x + len(query)], keep_internals=True).distance
        yield distance


if __name__ == '__main__':
    query = ["1", "3"]
    source = ["2", "3", "3", "1", "3", "2", "4", "6", "7", "9"]
    alignment1 = dtw(query, source, keep_internals=True)
    print(alignment1.distance)  # 2

    local_dtw_distance(query, source)

    # alignment4 = dtw(["1", "3"], ["3", "3", "1", "2", "2"], keep_internals=True)
    # print(alignment4.distance)  # 2
    #
    # alignment2 = dtw(["2", "3"], ["2", "3"], keep_internals=True)
    # print(alignment2.distance)  # 0
    #
    # alignment2 = dtw(["0"], ["2", "3", "3"], keep_internals=True)
    # print(alignment2.distance)
    # print(alignment1.distance - alignment2.distance)
    # # alignment3 = dtw(["2,4"], ["2", "3"], keep_internals=True)
    # print(alignment3.distance) # could not convert string to float: '2,4'
