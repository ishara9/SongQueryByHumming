import fnmatch
import os
import pickle
import re


def get_filtered_files(path, filters=['wav']):
    try:
        files = os.listdir(path)
        for filter_ in filters:
            for f in fnmatch.filter(files, "*.%s" % filter_):
                p = os.path.join(path, f)
                yield p
    except:
        print("File read error")


def pickle_data(data, file='data.pickle'):
    with open(file, 'wb') as outfile:
        pickle.dump(data, outfile)
        outfile.close()


def unpickle_data(file='data.pickle'):
    infile = open(file, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict


def file_pickle_rename(file_path):
    return file_path.replace("/", "_") + ".pickle"


def file_path_to_name_formatter(path):
    a = re.search(r'(\\)', path)
    file_name = path[a.start() + 1:-4]
    # print(file_name)
    return file_name


if __name__ == '__main__':
    file = "data/LocalHumData/sinhala"
    print(file_pickle_rename(file))
    filters = ['m4a']
    files = get_filtered_files(file, filters)
    for file_name in files:
        print(file_name)
