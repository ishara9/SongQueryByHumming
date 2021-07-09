from index import *
from logger import log_time


def evaluate_full():
    log_time("Start")

    file = "data/selected_set"
    create_data_model(file)
    model = unpickle_data(file=file_pickle_rename(file))
    log_time("UnPickled")
    # query_file = 'data/test/00020.wav'
    # query_file = 'data/test/happyBirthday_by_ishara.m4a'
    # query_file = 'data/test/nadi_ganga_hum.m4a'
    query_file = 'uploads/blob.wav'
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


def evaluate_partial():
    log_time("Start")

    file = "data/sinhala"

    model = unpickle_data(file=file_pickle_rename(file))
    log_time("UnPickled")
    filters = ['m4a']
    test_files = get_filtered_files('data/LocalHumData/sinhala', filters)
    tests_result = []
    actual_data = []
    for test_file in test_files:
        actual_data.append(test_file)
        query_data = create_query_data(test_file)
        _list = query(model, query_data)
        name, dis =_list[0]
        tests_result.append(name)

    # create_confusion_matrix(tests,actual_data)

    print('Predicted:' + str(tests_result))
    print('Actual data:' + str(actual_data))
    log_time("End")


def file_names_lister():
    file = "data/LocalHumData/sinhala"
    filters = ['m4a']
    files = get_filtered_files(file, filters)
    for file_name in files:
        print(file_name)


if __name__ == '__main__':
    evaluate_partial()
