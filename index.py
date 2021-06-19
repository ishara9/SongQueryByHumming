from file_processor import *
from logger import log_time
from audio_processor import *


def create_data_model(file, filters=['wav']):
    data_set = {}
    for file_name, _ in get_filtered_files(file, filters):
        pv = get_pitch_vector_by_file(file_name)
        data_set[file_name] = filter_outlier_pitches(pv)
    return data_set


def query(data_model, _query_pv):
    distance = {}
    for file_name, model_pv in data_model.items():
        distance[file_name] = calculate_dtw(model_pv, _query_pv)
    sorted_items = dict(sorted(distance.items(), key=lambda item: item[1])).items()
    first_item = list(sorted_items)[0]
    return first_item


if __name__ == '__main__':
    log_time()
    model = create_data_model('data/train')
    pickle_data(model)
    model = unpickle_data()
    test_file = 'data/test/nadi_ganga_hum.m4a'
    # test_file = 'data/test/happyBirthday_by_ishara.m4a'
    query_pv = get_pitch_vector_by_file(test_file)
    filtered_query_pv = filter_outlier_pitches(query_pv)
    name, dis = query(model, filtered_query_pv)

    print('The best match is:')
    print('  name:', name, ', distance:', dis)
    log_time()
