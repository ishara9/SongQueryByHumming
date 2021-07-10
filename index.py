from scipy.signal import medfilt

from audio_processor import *
from file_processor import *
from logger import log_time


def zero_index_dtw(filtered_notes):
    zero_index_array = np.asarray([0 for _ in range(len(filtered_notes))])
    variation = calculate_dtw(zero_index_array, filtered_notes)
    print(variation)
    return variation


def create_data_model(file, filters=['wav']):
    data_set = {}
    files = get_filtered_files(file, filters)
    for file_name in files:
        log_time("progress " + file_name)
        note_vector = get_note_vector_by_file(file_name)
        filtered_notes = filter_sound(note_vector)
        # zero_dtw = zero_index_dtw(filtered_notes)
        song = {'notes': filtered_notes, 'zero_dtw': 0}
        data_set[file_name] = song
    log_time("Model Created")
    pickle_data(data_set, file=file_pickle_rename(file))
    log_time("Pickled")
    return data_set


def filter_sound(notes):
    notes = medfilt(notes, 3)  # kernal size 3
    notes = filter_outlier_pitches(notes)
    # notes = np.gradient(notes)
    notes = np.diff(notes)
    return notes


def query(data_model, _query_pv):
    distance = {}
    for file_name, song1 in data_model.items():
        model_pv = song1['notes']
        zero_dtw = song1['zero_dtw']
        distance[file_name] = calculate_dtw(model_pv, _query_pv)
    sorted_items = dict(sorted(distance.items(), key=lambda item: item[1])).items()
    item_list = list(sorted_items)
    return item_list


def process_list(_list):
    for i, x in enumerate(_list):
        _list[i] = (file_path_to_name_formatter(x[0]), x[1])
    return _list


def search_tune():
    query_file = 'uploads/blob.wav'
    data_model = 'data/selected_set'
    _list = search_song(query_file, data_model)
    return process_list(_list[:10])


def search_song(query_file, data_model="data/sinhala"):
    log_time("Query start")
    model = unpickle_data(file=file_pickle_rename(data_model))
    log_time("UnPickled")
    query_data = create_query_data(query_file)
    _list = query(model, query_data)

    name, dis = _list[0]
    print(*_list, sep="\n")
    print('The best match is:')
    print('  name:', name, ', distance:', dis)
    log_time("Query End")
    return _list


def create_query_data(query_string):
    log_time("Query string processing :" + query_string)
    query_nv = get_note_vector_by_file(query_string)
    filtered_notes = filter_sound(query_nv)
    return filtered_notes


if __name__ == '__main__':
    log_time("Start")

    data_model = 'data/selected_set'
    create_data_model(data_model)

    query_file = 'data/test/nadi_ganga_hum.m4a'
    _list = search_song(query_file, data_model)

    log_time("End")
