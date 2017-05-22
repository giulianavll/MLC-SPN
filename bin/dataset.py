import numpy
import csv

INPUT_PATH = "data/"
OUTPUT_PATH = "data/queries/"

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
    narrayc = numpy.copy(narray)
    numpy.place(narrayc,numpy.logical_or(narrayc==-1,narrayc==-2), 2)
    dataset = numpy.copy(narrayc).astype(str)
    numpy.place(dataset,dataset=='2', '*')
    d=numpy.atleast_2d(dataset)
    numpy.savetxt(file_path, d, delimiter=sep, fmt='%s')
    return 
