from scipy.signal import medfilt

from audio_processor import *
from file_processor import *
from logger import log_time


def create_data_model(file, filters=['wav']):
    data_set = {}
    files = get_filtered_files(file, filters)
    for file_name in files:
        log_time("progress " + file_name)
        note_vector = get_note_vector_by_file(file_name)
        filtered_notes = filter_sound(note_vector)
        data_set[file_name] = filtered_notes
    log_time("Model Created")
    pickle_data(data_set, file=file_pickle_rename(file))
    log_time("Pickled")
    return data_set


def filter_sound(notes):
    notes = medfilt(notes)
    # notes = filter_outlier_pitches(notes)
    notes = np.gradient(notes)
    notes = np.diff(notes)
    return notes


def query(data_model, _query_pv):
    distance = {}
    for file_name, model_pv in data_model.items():
        distance[file_name] = calculate_dtw(model_pv, _query_pv)
    sorted_items = dict(sorted(distance.items(), key=lambda item: item[1])).items()
    item_list = list(sorted_items)
    return item_list


def process_list(_list, suffix):
    for i, x in enumerate(_list):
        _list[i] = (x[0][len("data/" + suffix + "\\"):], x[1])
    return _list


def search_tune():
    log_time("Start")
    # file = "dataSINHALA.pickle"
    file = "dataSELECTED_15SEC.pickle"
    model = unpickle_data(file=file)
    log_time("UnPickled")
    test_file = 'uploads/blob.wav'
    query_pv = get_note_vector_by_file(test_file)
    log_time("Query Audio")
    filtered_query_pv = filter_outlier_pitches(query_pv)
    note_diff = np.gradient(filtered_query_pv)
    _list = query(model, note_diff)
    name, dis = _list[0]
    print('The best match is:')
    print('  name:', name, ', distance:', dis)
    log_time("End")
    _list[:10]
    return process_list(_list[:10], "SELECTED_SET")


def create_query_data(query_string):
    log_time("Query string processing :" + query_string)
    query_nv = get_note_vector_by_file(query_string)
    filtered_notes = filter_sound(query_nv)
    return filtered_notes


if __name__ == '__main__':
    log_time("Start")

    file = "data/sinhala"
    create_data_model(file)
    model = unpickle_data(file=file_pickle_rename(file))
    log_time("UnPickled")
    # query_file = 'data/test/00020.wav'
    # query_file = 'data/test/happyBirthday_by_ishara.m4a'
    # query_file = 'data/test/nadi_ganga_hum.m4a'
    query_file = 'data/LocalHumData/sinhala/adara mal wala.m4a'
    # query_file = 'uploads/blob.wav'
    # query_file = 'data/test/nadi_ganga_hum.m4a'
    # query_file = 'data/test/2000_copy_mono.wav'
    # query_file = 'data/test/Happy_bday_long.m4a'

    query_data = create_query_data(query_file)

    _list = query(model, query_data)
    name, dis = _list[0]

    print(_list)
    print('The best match is:')
    print('  name:', name, ', distance:', dis)
    log_time("End")
