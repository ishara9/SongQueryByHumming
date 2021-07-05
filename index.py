from file_processor import *
from logger import log_time
from audio_processor import *


def create_data_model(file, filters=['wav']):
    data_set = {}
    files = get_filtered_files(file, filters)
    for file_name in files:
        log_time("progress " + file_name)
        pv = get_pitch_vector_by_file(file_name)
        data_set[file_name] = filter_outlier_pitches(pv)
    return data_set


def query(data_model, _query_pv):
    distance = {}
    for file_name, model_pv in data_model.items():
        distance[file_name] = calculate_dtw(model_pv, _query_pv)
    sorted_items = dict(sorted(distance.items(), key=lambda item: item[1])).items()
    item_list = list(sorted_items)
    return item_list


def process_list(_list):
    for i, x in enumerate(_list):
        _list[i] = (x[0][len("data/mix100\\"):], x[1])
    return _list


def search_tune():
    log_time("Start")
    file = "dataMIX100.pickle"
    model = unpickle_data(file=file)
    log_time("UnPickled")
    test_file = 'uploads/blob.wav'
    query_pv = get_pitch_vector_by_file(test_file)
    log_time("Query Audio")
    filtered_query_pv = filter_outlier_pitches(query_pv)
    _list = query(model, filtered_query_pv)
    name, dis = _list[0]
    print('The best match is:')
    print('  name:', name, ', distance:', dis)
    log_time("End")
    _list[:10]
    return process_list(_list[:10])


if __name__ == '__main__':
    log_time("Start")
    # file = "dataMIX100.pickle"
    file = "dataSINHALA.pickle"

    # model = create_data_model('data/test')
    # model = create_data_model('data/sinhala')
    # model = create_data_model('data/mix100')
    log_time("Model Created")

    # pickle_data(model, file=file)
    log_time("Pickled")
    model = unpickle_data(file=file)
    log_time("UnPickled")
    # test_file = 'data/test/00020.wav'
    # test_file = 'data/test/happyBirthday_by_ishara.m4a'
    test_file = 'data/test/anatha_maruthe.m4a'
    # test_file = 'data/test/nadi_ganga_hum.m4a'
    # test_file = 'data/test/2000_copy_mono.wav'
    # test_file = 'data/test/Happy_bday_long.m4a'
    query_pv = get_pitch_vector_by_file(test_file)
    log_time("Query Audio")
    filtered_query_pv = filter_outlier_pitches(query_pv)
    name, dis = query(model, filtered_query_pv)[0]

    print('The best match is:')
    print('  name:', name, ', distance:', dis)
    log_time("End")
