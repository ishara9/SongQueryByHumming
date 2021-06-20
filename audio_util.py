from collections import Iterable

import numpy as np


def sliding_window(sequence, window_size=50, shift_ratio=1):
    shift = int(shift_ratio * window_size)

    validate_sliding_window_parameters(sequence, shift, window_size)

    frame_count = ((len(sequence) - window_size) / shift) + 1

    for i in range(0, int(frame_count) * shift, shift):
        yield sequence[i:i + window_size]


def validate_sliding_window_parameters(sequence, shift, window_size):
    assert isinstance(window_size, int) and isinstance(shift, int)
    assert shift <= window_size
    assert window_size <= len(sequence)
    assert isinstance(sequence, Iterable)


def filter_outlier_pitches(pv, threshold=10):
    adjusted = pv - np.median(pv)
    loc = (abs(adjusted) > threshold)
    pv[loc] = 0
    return pv
