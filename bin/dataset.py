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
    r = list(reader)
    # *(-1):unknow value, %(-2): MPE query value
    #vals = [-1 if x=='*' else -2 if x=='%' else x for line in r for x in line]
    vals = r
    dataset = numpy.array(vals).astype(type)
    return dataset


def numpy_2_file(narray, file, path=OUTPUT_PATH, sep=',' ):
    """
    convert numpy into csv file , for ID(libra) algotihms
    """
    file_path = path + file
    vals = ['*' if x==-1 else '*' if x==-2 else x for line in narray for x in line]
    dataset = numpy.array(vals).astype(str)
    numpy.savetxt(file_path, dataset, delimiter=sep, fmt='%s')
    return 
