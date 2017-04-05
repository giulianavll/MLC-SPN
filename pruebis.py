from bin.classifier import *
from bin import dataset 

train = dataset.csv_2_numpy('testm'+'.train')
c= MClassifierCCG('ac',train,'testm', 3,2,False,'ccg')
#Train a classifier
#c.create_classifier()
#Classify and get files with the results for Hamming Score, Exact match and  Accuracy 
test = dataset.csv_2_numpy('testm'+'.test')
c.classify_metrics(test)
