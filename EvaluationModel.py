from ConfusionMatrix import *
from index import *
from logger import log_time


def evaluate_model():
    log_time("Start")

    file = "data/selected_set"

    model = unpickle_data(file=file_pickle_rename(file))
    log_time("UnPickled")
    filters = ['m4a']
    test_files = get_filtered_files('data/LocalHumData/sinhala', filters)
    tests_result = {}
    actual_data = []
    for test_file in test_files:
        actual_data.append(file_path_to_name_formatter(test_file))
        query_data = create_query_data(test_file)
        _list = query(model, query_data, False)
        temp = []
        for i, predict in enumerate(_list[:5]):
            name, dis = predict
            temp.append(file_path_to_name_formatter(name))
        tests_result[test_file] = temp

    print('Predicted:' + str(tests_result))
    print('Actual data:' + str(actual_data))

    tests, actual = format_test_results(tests_result)
    # actual = format_actual_results(actual_data)
    testable_values = set(actual.copy())
    testable_values.update(set(tests.copy()))
    # testable_values.add('no_match')
    ordered = list(testable_values)

    tests.extend(ordered)
    actual.extend(ordered)

    print('Formatted Predicted:' + str(tests))
    print('Formatted Actual data:' + str(actual))
    print('Formatted Testable values:' + str(testable_values))

    create_confusion_matrix(tests, actual, list(sorted(testable_values)))
    get_classification_report(tests, actual, list(sorted(testable_values)))

    log_time("End")


def equals(string1, string2):
    return string1.lower() == string2.lower()


def format_test_results(tests_result):
    single_key = []
    single_value = []

    for key in tests_result:
        value = tests_result[key]
        is_contains = False
        formatted_key = file_path_to_name_formatter(key)
        single_key.append(formatted_key)
        for each in value:
            if equals(formatted_key, each):
                is_contains = True
                continue
        if is_contains:
            single_value.append(formatted_key)
        else:
            single_value.append(value[0])

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
