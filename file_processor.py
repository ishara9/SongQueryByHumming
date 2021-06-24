import fnmatch
import os
import pickle


def get_filtered_files(path, filters):
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
