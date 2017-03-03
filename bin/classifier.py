
import numpy as np
from math import *
from numpy.testing import assert_almost_equal
from bin.spn import *


class Classifier(object):
	
	def __init__(self,
				 learn_algorithm, 
				 train,
				 name,
				 nlabels,
				 nattributes):
		self.learn_algorithm = learn_algorithm
		self.train = train
		self.nlabels = nlabels
		self.nattributes	= nattributes
		self.name = name
		
	def generate_SPNID(self,
					   subset,
					   name):
		aspn = SPNID(subset,name)
		return 	aspn.learn() 	

	def generate_SPNAL(self
					   subset,
					   name):
		aspn = SPNAL(subset,name)
	 	return aspn.learn()

	def generate_SPNAC(self
					   subset,
					   name):
		aspn = SPNAC(subset, name)
	 	return aspn.learn()	

	def generate_order(self):
		#default Coordinates order
		#see implementation on BRApproach and CCApproach
		self.order=[i for i in range(np.shape(self.train)[0])]

	def preprocess(self):
		#implemented in the specific a
		pass

	def create_classifier(self):
		#Create classifier, learning SPNs with the algorthm of the params
		self.options = {
		'id' : self.generate_classifierID,
		'al' : self.generate_classifierAL,
		'ac' : self.generate_classifierAC
		}
		self.generate_order()
		self.models=[]

		subsets = self.preprocess(order)
		method_learn = self.options[self.learn_algorithm]
		for i,s in enumerate(subsets):
			if i==0 and len(subsets) == 1:
				name_spn = ''
			else:
				name_spn = 'L'+i 		
			models.append(method_learn(s, self.name +name_spn ))

	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass


class MClassifierBR(classifier):
	def __init__(self,learn_algorithm, train):
		 super().__init__(learn_algorithm, train)

	def preprocess(self):
		#BR preprocess

	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass

class MClassifierCCG(classifier):
	def __init__(self,learn_algorithm, train):
		 super().__init__(learn_algorithm, train)

	def preprocess(self):
		subsets = []
		
		dataset = self.train
		v_order = numpy.zeros(dataset.shape())
		for l in range (0,self.nlabels): 
			tlist = []
			c = self.order[l]
        	instance_t = numpy.zeros(l+1+self.nattributes)
        	for i_ds , i_vo in zip(dataset,v_order):

        		
        		if l != 0 :
        			instance_t[]
        		instance_t[0:l]=instance_t[0:]
        		instance_t[l]=instance_t[c]
        		instance_t[l+1:]=instance[self.nlabels:]





            	instance_t[0:l+1]=instance[0:c+1]
            	instance_t[l+1:]=instance[n_labels:]
            	tlist.append(instance_t)
        	subsets.append(numpy.array(tlist).astype(int))
        	dataset=subsets[l]
		return subsets

	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass

class MClassifierCC1(classifier):
	def __init__(self,learn_algorithm, train):
		 super().__init__(learn_algorithm, train)

	def preprocess(self):
		subsets=[]
		subsets.append(self.train)
		return subsets


	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass

class MClassifierMPE(classifier):
	def __init__(self,learn_algorithm, train):
		 super().__init__(learn_algorithm, train)

	def preprocess(self):
		subsets=[]
		subsets.append(self.train)
		return subsets


	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass

class MClassifierLP(classifier):
	def __init__(self,learn_algorithm, train):
		 super().__init__(learn_algorithm, train)

	def preprocess(self):
		subsets=[]
		subsets.append(self.train)
		return subsets


	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass


