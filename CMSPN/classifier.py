
import numpy as np
from math import *
from numpy.testing import assert_almost_equal
from spn import SPNID
from spn import SPNAC
from spn import SPNAL
from libra.bin import learn_IDSPN
import datetime


class Classifier(object):
	
	def __init__(self,
				 learn_algorithm, 
				 train,
				 name,
				 nlabels,
				 natributes):
		self.learn_algorithm = learn_algorithm
		self.train = train
		self.nlabels = nlabels
		self.natributes	= natributes
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
		pass

	def create_classifier(self):
		self.options = {
		'id' : self.generate_classifierID,
		'al' : self.generate_classifierAL,
		'ac' : self.generate_classifierAC
		}
		self.generate_order()
		self.models=[]
		subsets = self.preprocess(order)
		method_learn = self.options[self.learn_algorithm]
		i = 1
		for x in subsets:		
			models.append(method_learn(x, self.name +'L'+i ))
			i++

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
		#BR preprocess

	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass

class MClassifierCC1(classifier):
	def __init__(self,learn_algorithm, train):
		 super().__init__(learn_algorithm, train)

	def preprocess(self):
		#BR preprocess

	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass

class MClassifierMPE(classifier):
	def __init__(self,learn_algorithm, train):
		 super().__init__(learn_algorithm, train)

	def preprocess(self):
		#BR preprocess

	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass

class MClassifierLP(classifier):
	def __init__(self,learn_algorithm, train):
		 super().__init__(learn_algorithm, train)

	def preprocess(self):
		#BR preprocess

	def classify(self,
				 evidence,
				 query,
				 real_values):
		pass


