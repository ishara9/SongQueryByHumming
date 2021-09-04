from scipy.signal import medfilt
from tornado import concurrent
from audio_processor import *
from file_processor import *
from logger import log_time


def zero_index_dtw(filtered_notes, white_noise):
    # white_noise = np.asarray([0 for _ in range(len(filtered_notes))])
    variation = calculate_dtw(white_noise, filtered_notes)
    print(variation)
    return variation


def create_data_model(file, filters=['wav']):
    data_set = {}
    files = get_filtered_files(file, filters)
    white_noise = get_white_noise_data()
    for file_name in files:
        log_time("progress " + file_name)
        note_vector = get_note_vector_by_file(file_name)
        filtered_notes = filter_sound(note_vector)
        zero_dtw = zero_index_dtw(filtered_notes, white_noise)
        song = {'notes': filtered_notes, 'zero_dtw': zero_dtw}
        data_set[file_name] = song
    log_time("Model Created")
    pickle_data(data_set, file=file_pickle_rename(file))
    log_time("Pickled")
    return data_set


def filter_sound(notes):
    notes = medfilt(notes, 3)  # kernal size 3
    notes = filter_outlier_pitches(notes)
    # notes = np.gradient(notes)
    log_time("np.diff Start")
    notes = np.diff(notes)
    log_time("np.diff End")
    return notes


def get_white_noise_data():
    query_file = 'data/white_noise.wav'
    query_data = create_query_data(query_file)
    return query_data


def distance_calculator_worker(item, _query_pv, zero_remove):
    distance = {}
    file_name = item[0]
    song1 = item[1]
    model_pv = song1['notes']
    zero_dtw = song1['zero_dtw']
    log_time("data_model:loop file_name:" + str(file_name))
    if zero_remove:
        distance[file_name] = calculate_dtw(model_pv, _query_pv) - zero_dtw
    else:
        distance[file_name] = calculate_dtw(model_pv, _query_pv)
    log_time("data_model:loop distance:" + str(distance[file_name]))
    return distance


def query(data_model, _query_pv, zero_remove=False):
    log_time("query start")
    distance = {}
    log_time("data_model:loop start")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(distance_calculator_worker, item, _query_pv, zero_remove) for item in
                   data_model.items()]
        for future in concurrent.futures.as_completed(futures):
            distance.update(future.result())

    log_time("data_model:loop end")
    sorted_items = dict(sorted(distance.items(), key=lambda item: item[1])).items()
    item_list = list(sorted_items)
    log_time("query end")
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
    log_time("search_song start")
    model = unpickle_data(file=file_pickle_rename(data_model))
    log_time("UnPickled")
    query_data = create_query_data(query_file)
    _list = query(model, query_data, False)

    name, dis = _list[0]
    print(*_list, sep="\n")
    print('The best match is:')
    print('  name:', name, ', distance:', dis)
    log_time("Query End")
    return _list


def create_query_data(query_string):
    log_time("============= Query string processing :" + query_string + "=============")
    query_nv = get_note_vector_by_file(query_string)
    filtered_notes = filter_sound(query_nv)
    log_time("create_query_data End")
    return filtered_notes


if __name__ == '__main__':
    log_time("Start")

    data_model = 'data/selected_set'
    create_data_model(data_model)

    query_file = 'data/test/alay ish B.m4a'
    _list = search_song(query_file, data_model)

    log_time("End")
