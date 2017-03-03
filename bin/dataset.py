import numpy
import csv

INPUT_PATH = "data/"


def csv_2_numpy(file, path=INPUT_PATH, sep=',', type='int8'):
    """
    convert csv into numpy
    """
    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = numpy.array(x).astype(type)
    return dataset

def numpy_2_file(narray, file, path=INPUT_PATH, sep=',', ):
	"""
    convert numpy into csv file
    """
    file_path = path + file
    np.savetxt(file_path, narray, delimiter=sep, fmt='%i')
    return 
