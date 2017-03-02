import argparse
import dataset
from classifier import MClassifierBR
from classifier import MClassifierCCG
from classifier import MClassifierCC1
from classifier import MClassifierLP
from classifier import MClassifierMPE

#########################################
# creating the opt parser

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', '--dataset', type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. flags0)')
parser.add_argument('-ap','--approach' ,type=str, nargs='?',
                    default='cc',
                    help='Multilabel Classification Approach br, cc, cc1, mpe, lp ')
parser.add_argument('-ml', '--mlearning',type=str, nargs='?',
                    default='al',
                    help='Method for learning SPN id, al, ac')

parser.add_argument('-nl', '--nLabels', type=int, nargs='?',
                    default=1,
                    help='Number of labels in dataset')

parser.add_argument('-nat', '--nAtrb', type=int, nargs='?',
                    default=1,
                    help='Number of atributs in dataset')
parsing the args
args = parser.parse_args()

dataset = args.dataset 
spn_mlearn = args.mlearning
approach = args.approach
n_labels = args.nLabels
n_attributes = args.nAtrb

train = dataset.csv_2_numpy(dataset + ".train")

if approach == 'br':
    c= MClassifierBR(spn_mlearn,
                     train,
                     dataset,
                     n_labels,
                     n_attributes)
elif approach == 'cc':
    c= MClassifierCCG(spn_mlearn,
                     train,
                     dataset,
                     n_labels,
                     n_attributes)
elif approach == 'cc1':
    c= MClassifierCC1(spn_mlearn,
                     train,
                     dataset,
                     n_labels,
                     n_attributes)
elif approach == 'mpe':
    c= MClassifierMPE(spn_mlearn,
                     train,
                     dataset,
                     n_labels,
                     n_attributes)
elif approach == 'lp':
    c= MClassifierLP(spn_mlearn,
                     train,
                     dataset,
                     n_labels,
                     n_attributes)
c.create_classifier()
c.get_metrics()