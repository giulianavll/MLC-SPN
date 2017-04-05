import numpy
import csv

INPUT_PATH = "data/"
OUTPUT_PATH = "data/queries"

def csv_2_numpy(file, path=INPUT_PATH, sep=',', type='int8'):
    """
    convert csv into numpy
    """
    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = numpy.array(x).astype(type)
    return dataset


def numpy_2_file(narray, file, path=OUTPUT_PATH, sep=',' ):
    """
    convert numpy into csv file , for ID(libra) algothim
    """
    file_path = path + file
    dataset = numpy.copy(narray).astype(str)
    numpy.place(dataset,numpy.logical_or(dataset=='-1',dataset=='-2'), '*')
    numpy.savetxt(file_path, dataset, delimiter=sep, fmt='%s')
    return 
