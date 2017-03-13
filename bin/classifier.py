import numpy 
from math import *
from numpy.testing import assert_almost_equal
from bin.spn import *
import operator
import logging
import itertools



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


	def generate_SPNAL(self,
					   subset,
					   name):
		aspn = SPNAL(subset,name)
		return aspn.learn()

	def generate_SPNAC(self,
					   subset,
					   name):
		aspn = SPNAC(subset, name)
		return aspn.learn()	

	def generate_order(self):
		#default Coordinates order
		#see implementation on BRApproach and CCApproach
		self.order=[i for i in range(numpy.shape(self.train)[0])]

	def preprocess(self):
		#implemented in the specific class
		pass

	def create_classifier(self):
		#Create classifier, learning SPNs with the algorthm of the params
		self.options = {
		'id' : self.generate_SPNID,
		'al' : self.generate_SPNAL,
		'ac' : self.generate_SPNAC
		}
		self.generate_order()
		self.models=[]
		subsets = self.preprocess()
		method_learn = self.options[self.learn_algorithm]
		for i,s in enumerate(subsets):
			if i==0 and len(subsets) == 1:
				name_spn = ''
			else:
				name_spn = 'L'+str(i) 		
			self.models.append(method_learn(s, self.name +name_spn ))

	def get_metrics(self, teste):

		d_predict = self.classify_batch(teste)
		self.generate_metrics(teste,d_predict,self.name)
		return 

	def classify_batch(self,
		               teste):
		pass

	def classify_qev(query, nlabel, a_spn ):
		l=self.order
		nmodel = 0
		log_proba = a_spn.query_PC(ev,query)
		classify =[]
		for i,lp in enumerate(log_proba):
			if lp < log(0.45):
				val =query[i][l]
				if val == 1:
					query[i][l]=0
				else:
					query[i][l]=1
			classify.append(query[i])
		return classify


class MClassifierBR(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes)

	def preprocess(self):
		#BR preprocess
		logging.info('-- BR Preprocess --')
		subsets = []
		dataset = self.train
		for l in range (0,self.nlabels):
			tlist = []
			instance_t = numpy.zeros(l+1+self.nattributes)
			for instance in dataset:
				instance_t[:l+1]=instance[:l+1]
				instance_t[l+1:]=instance[self.nlabels:]
				tlist.append(instance_t)
			subsets.append(numpy.array(tlist).astype(int))
		return subsets


	def classify_batch(self,
				 teste):
		pass

class MClassifierCCG(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes)

	def reorderingDS(self,dataset):
		for i,instance in enumerate(dataset):
			ix , tuple_o = zip(*sorted(zip(self.order, instance),  key=operator.itemgetter(0), reverse=False))
			inst_o = numpy.asarray(tuple_o,dtype=numpy.int32)
			dataset[i] = inst_o
		return dataset



	def preprocess(self):
		logging.info('-- CC Preprocess --')
		subsets = []
		dataset = self.train
		for l in range (0,self.nlabels): 
			tlist = []
			instance_t = numpy.zeros(l+1+self.nattributes)
			#Reordering in training set given the label order
			if (l==0):
				dataset= self.reorderingDS(dataset)
			#Preprocessing
			for instance in dataset:
				instance_t[:l+1]=instance[:l+1]
				instance_t[l+1:]=instance[self.nlabels:]
				tlist.append(instance_t)
			subsets.append(numpy.array(tlist).astype(int))
		return subsets

	def generate_EQ(self, nlabel,predict):
		q =[]
		e =[]
		l= self.order[nlabel]
		for instance in predict:
			if(nlabel ==0):
				n = self.nlabels - 1
				unknown_v = "".join(-1 for i in range(n))
				instance_t[nlabels+1 : self.nlabels] = unknown_v
				q_instance = instance_t
				e_instance = instance_t
			q_instance[l] = '1'  #query value P(Li=1)
			e_instance[l] = '*'
		return e, q


	def undo_OrderingDS(self,dataset):
		v_coo = list(range(0,len(self.order)))
		i , tuple_no = zip(*sorted(zip(self.order, v_coo),  key=operator.itemgetter(0), reverse=False))
		n_order = list(tuple_no)
		for i,instance in enumerate(dataset):
			ix , tuple_o = zip(*sorted(zip(n_order, instance),  key=operator.itemgetter(0), reverse=False))
			inst_o = numpy.asarray(tuple_o,dtype=numpy.int32)
			dataset[i] = inst_o
		return dataset


	def classify_batch(self,teste):
		#predict = reorderingDS(teste)
		for i in range(0,len(self.nlabels)):
			(ev , query) = generate_EQ(i,predict)
			predict = classify_qev(ev , query , i, self.models[i])
		#predict = undo_OrderingDS(predict)	
		return predict	



class MClassifierCC1(MClassifierCCG):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes)

	def preprocess(self):
		logging.info('-- CC1 Preprocess --')
		subsets=[]
		subsets.append(self.train)
		return subsets
	 

	def classify_batch(self,teste):		
		for i in range(0,len(self.nlabels)):
			(ev , query) = generate_EQ(i,predict)
			predict = classify_qev(ev , query , i, self.models[0])
		return predict	


class MClassifierMPE(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes)

	def preprocess(self):
		logging.info('-- MPE Preprocess --')
		subsets=[]
		subsets.append(self.train)
		return subsets


	def classify_batch(self, teste):
		pass

class MClassifierLP(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes)

	def preprocess(self):
		logging.info('-- LP Preprocess --')
		subsets=[]
		subsets.append(self.train)
		return subsets


	def classify_batch(self,teste):
		power_set = list(itertools.product([0, 1], repeat = self.nlabels))
		new_teste=[]
		for instance in teste:
			q_instance = instance
			e_instance = instance
			for v_labels in power_set:
				q_instance[:self.nlabels] = v_labels
				#q_evidence[:self.nlabels] = 
				q_teste.append(n_instance)
		n_teste = numpy.array(nl_teste).astype(int)




