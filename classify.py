import argparse
import numpy
from bin import dataset as dataset_m
from bin.classifier import *
import logging
#########################################
# creating the opt parser

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', '--dataset', type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. flags0)')
parser.add_argument('-ap','--approach' ,type=str, nargs='?',
                    default='cc1',
                    help='Multilabel Classification Approach br, cc, cc1, mpe, lp ')
parser.add_argument('-ml', '--mlearning',type=str, nargs='?',
                    default='al',
                    help='Method for learning SPN id, al, ac')

parser.add_argument('-nl', '--nLabels', type=int, nargs='?',
                    default=1,
                    help='Number of labels in dataset')

parser.add_argument('--folds', action='store_true', default = False )
parser.add_argument('--bagg', action='store_true', default = False )
# parsing the args
args = parser.parse_args()

dataset = args.dataset[0] 
spn_mlearn = args.mlearning
approach = args.approach
n_labels = args.nLabels
folds = args.folds
bgg = args.bagg

print('-- Init the TRAINING --')
#load the training set
print(folds)

if(folds):
    val = 5
else:
    train = dataset_m.csv_2_numpy(dataset+'.train')
    val = 0
    dataset_f=dataset

for f in range(0,val):
    if val == 5:
        print(dataset+str(f)+'.train')
        dataset_f=dataset+str(f)
    train= dataset_m.csv_2_numpy(dataset_f+'.train')
    y = train.shape[1]
    n_attributes = y-n_labels
    test =dataset_m.csv_2_numpy(dataset_f+'.test')
    #choose the approach for MLC
    logging.info('Init Classification Generation')
    if approach == 'br':
        c= MClassifierBR(spn_mlearn,
                         train,
                         dataset_f,
                         n_labels,
                         n_attributes,
                         bgg)
    elif approach == 'cc':
        c= MClassifierCCG(spn_mlearn,
                         train,
                         dataset_f,
                         n_labels,
                         n_attributes,
                         bgg)
    elif approach == 'pcc':
        c= MClassifierPCC(spn_mlearn,
                         train,
                         dataset_f,
                         n_labels,
                         n_attributes,
                         bgg)
    elif approach == 'mpe':
        c= MClassifierMPE(spn_mlearn,
                         train,
                         dataset_f,
                         n_labels,
                         n_attributes,
                         bgg)
    elif approach == 'lp':
        c= MClassifierLP(spn_mlearn,
                         train,
                         dataset_f,
                         n_labels,
                         n_attributes,
                         bgg)

    #Train a classifier
    c.create_classifier()

    #Classify and get files with the results for Hamming Score, Exact match and  Accuracy 
    c.classify_metrics(test)