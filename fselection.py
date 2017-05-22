import numpy as np
import csv
from sklearn.feature_selection import chi2, SelectKBest

INPUT_PATH = "data/"
OUTPUT_PATH="data/"
selected_features = []
nlabels=30 
ndataset = 'business'
fold = '4'


def csv_2_numpy(file, path=INPUT_PATH, sep=',', type='int8'):
    """
    convert csv into numpy
    """
    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = np.array(x).astype(type)
    return dataset
dataset = csv_2_numpy(ndataset+fold+'.train')
dt= csv_2_numpy(ndataset+fold+'.test')

def createDataset(sfeats,name):
    print(name)
    print(dataset.shape[0])
    print(dt.shape[0])
    new_dst = np.zeros((dataset.shape[0],sfeats.shape[0]+nlabels),dtype=np.int8)
    new_ds = np.zeros((dt.shape[0],sfeats.shape[0]+nlabels),dtype=np.int8)
    new_dst[:,:nlabels] = dataset[:,:nlabels]
    new_ds[:,:nlabels] = dt[:,:nlabels]
    for i,idx in enumerate(sfeats):
        new_dst[:,i+nlabels]=dataset[:,idx+nlabels]
        new_ds[:,i+nlabels]=dt[:,idx+nlabels]
    print(new_ds.shape)
    print(new_dst.shape)   
    file_path = OUTPUT_PATH+name
    np.savetxt(file_path+'.train', new_dst, delimiter=',', fmt='%d')
    np.savetxt(file_path+'.test', new_ds, delimiter=',', fmt='%d')

X=dataset[:,nlabels:]
for label in range(nlabels):
	Y=dataset[:,label]
	selector = SelectKBest(chi2, k='all')
	selector.fit(X, Y)
	sc = np.asarray(list(selector.scores_))
	#print('label'+str(label))
	#print(sc)
	selected_features.append(sc)
#MeanCS
mean_l = np.nanmean(selected_features, axis=0) 
mean_l[np.isnan(mean_l)]=0
feautures=[]
features60_mean = (mean_l> np.percentile(mean_l,40)).nonzero()[0]
features80_mean = (mean_l> np.percentile(mean_l,20)).nonzero()[0]

createDataset(features60_mean,ndataset+'60mean'+fold)
createDataset(features80_mean,ndataset+'80mean'+fold)

