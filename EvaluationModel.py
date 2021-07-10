from ConfusionMatrix import *
from index import *
from logger import log_time


def evaluate_model():
    log_time("Start")

    file = "data/sinhala"

    model = unpickle_data(file=file_pickle_rename(file))
    log_time("UnPickled")
    filters = ['m4a']
    test_files = get_filtered_files('data/LocalHumData/sinhala', filters)
    tests_result = {}
    actual_data = []
    for test_file in test_files:
        test_name = file_path_to_name_formatter(test_file)
        actual_data.append(test_name)
        query_data = create_query_data(test_file)
        _list = query(model, query_data)
        temp = []
        for i, predict in enumerate(_list[:5]):
            name, dis = predict
            temp.append(file_path_to_name_formatter(name))
        tests_result[test_name] = temp

    print('Predicted:' + str(tests_result))
    print('Actual data:' + str(actual_data))

    tests, actual = format_test_results(tests_result)
    # actual = format_actual_results(actual_data)
    testable_values = tests.copy()
    testable_values.append('no_match')

    print('Formatted Predicted:' + str(tests))
    print('Formatted Actual data:' + str(actual))
    print('Formatted Testable values:' + str(testable_values))

    create_confusion_matrix(tests, actual, testable_values)
    get_classification_report(tests, actual, testable_values)

    log_time("End")


def equals(string1, string2):
    return string1.lower() == string2.lower()


def format_test_results(tests_result):
    single_key = []
    single_value = []

    for key in tests_result:
        value = tests_result[key]
        is_contains = False
        single_key.append(key)
        for each in value:
            if equals(key, each):
                is_contains = True
                continue
        if is_contains:
            single_value.append(key)
        else:
            single_value.append('no_match')

    return single_key, single_value


def format_actual_results(actual_data):
    return actual_data


def file_names_lister():
    file = "data/LocalHumData/sinhala"
    filters = ['m4a']
    files = get_filtered_files(file, filters)
    for file_name in files:
        print(file_name)


if __name__ == '__main__':
    evaluate_model()
    # format_test_results([])
