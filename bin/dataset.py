import numpy
import csv

DATA_PATH = "dataset/"

def csv_2_numpy(file, path=DATA_PATH, sep=',', type='int8'):
    """
    WRITEME
    """
    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = numpy.array(x).astype(type)
    return dataset
