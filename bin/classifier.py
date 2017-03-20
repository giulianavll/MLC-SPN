import numpy 
from math import *
from numpy.testing import assert_almost_equal
from bin.spn import *
import operator
import logging
import itertools
import csv
import random
import queue 


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
		self.nattributes = nattributes
		self.name = name
		
	def generate_SPNID(self,
					   subset,
					   name):
		aspn = SPNID(subset,name)
		aspn.learn() 	
		return 	aspn


	def generate_SPNAL(self,
					   subset,
					   name):
		aspn = SPNAL(subset,name)
		aspn.learn()
		return aspn

	def generate_SPNAC(self,
					   subset,
					   name):
		aspn = SPNAC(subset, name)
		aspn.learn()	
		return aspn

	def generate_order(self):
		#default Coordinates order
		#see implementation on BRApproach and CCApproach
		order=[i for i in range(self.nlabels)]
		random.shuffle(order)
		self.order=order


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
		
		self.models=[]
		subsets = self.preprocess()
		method_learn = self.options[self.learn_algorithm]
		for i,s in enumerate(subsets):
			if i==0 and len(subsets) == 1:
				name_spn = ''
			else:
				name_spn = 'L'+str(i) 		
			self.models.append(method_learn(s, self.name +name_spn ))

	def load_variance(self):
		range_var = []
		#load range de number of labels from file 
		drct = "results/variance/"+self.name
		if os.path.exists(drct):
			reader = csv.reader(open(drct, "r"), delimiter=',')
			range_var =  list(map(int,list(reader)[0]))
		range_var.sort()
		return range_var

	def get_metrics(self, teste):

		d_predict = self.classify_batch(teste)
		self.generate_metrics(teste,d_predict,self.name)
		return 

	def classify_batch(self,
		               teste):
		pass

	def classify_qev(ev, query, nl, a_spn ):
		if self.order:
			l=self.order[nl]
		else:
			l=nl
		q_pr=queue.PriorityQueue()
		log_proba = a_spn.query_PC(ev,query)
		for i,lp in enumerate(log_proba):
			if lp < log(0.5):
				val =query[i][l]
				if val == 1:
					query[i][l] = 0
					if self.order:
						tup = (-lp,i,l)
						q_pr.put(tup)
				else:
					query[i][l]=1
		if self.order:
			lq_prob.append(q_pr)
		return query


class MClassifierBR(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes)

	def preprocess(self):
		#BR preprocess Generate l subsets for each label-classifier 
		logging.info('-- BR Preprocess --')
		subsets = []
		dataset = self.train
		for l in range (0,self.nlabels):
			tlist = []
			instance_t = numpy.zeros(1+self.nattributes)
			for instance in dataset:
				instance_t[0]=instance[l]
				instance_t[1:]=instance[self.labels:]
				tlist.append(instance_t)
			subsets.append(numpy.array(tlist).astype(int))
		return subsets

	def generate_EQ(self,l,test):
		q =[]
		e =[]
		unknown_v = -1
		for instance in test:
			instance_t = numpy.zeros(1+self.nattributes)
			instance_t[1:] = instance[self.nlabels:]
			q_instance = instance_t
			e_instance = instance_t
			q_instance[0] = 1  #query value P(Li=1)
			e_instance[0] = -1
		return e, q

	def classify_batch(self, test):
		for i in range(0,len(self.nlabels)):
			(ev , query) = generate_EQ(i,test)
			predict = classify_qev(ev , query , 0, self.models[i])
			#capture and concatenate predictions ?????????
		#predict = undo_OrderingDS(predict)	
		return predict	



class MClassifierCCG(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		super().__init__(learn_algorithm, train,name,nlabels,nattributes)
		self.generate_order()
		self.lq_prob = [] 

	def reorderingDS(self,dataset):
		datat=dataset.transpose()
		copyd= numpy.copy(datat)
		for l in range(self.nlabels):
			datat[l]=copyd[self.order[l]]
		dataset=datat.transpose()
		return dataset


	def undo_OrderingDS(self,dataset):
		datat=dataset.transpose()
		copyd= numpy.copy(datat)
		for l in range(self.nlabels):
			datat[self.order[l]]=copyd[l]
		dataset=datat.transpose()
		return dataset
		

	def preprocess(self):
		logging.info('-- CC Preprocess --')
		subsets = []
		dataset = self.train
		#Reordering in training set given the label order
		dataset= self.reorderingDS(dataset)
		for l in range (self.nlabels): 
			t_list = []
			instance_t = numpy.zeros(l+1+self.nattributes)			
			#Preprocessing
			for instance in dataset:
				instance_t[:l+1]=instance[:l+1]
				instance_t[l+1:]=instance[self.nlabels:]
				t_list.append(instance_t)
			subsets.append(numpy.array(t_list).astype(int))
		return subsets

	def generate_EQ(self, nlabel,predict):
		q =[]
		e =[]
		#l= self.order[nlabel]
		n = self.nlabels 
		unknown_v = []
		instance_t = numpy.zeros(nlabel+1+self.nattributes)			
		for instance in predict:
			if nlabel==0:
				instance_t[nlabel+1:] = instance[self.nlabels:]
			else:
				instance_t[0:nlabel] = instance[0:nlabel]
				instance_t[nlabel+1:]=instance[nlabel:]
			q_instance = numpy.copy(instance_t)
			e_instance = numpy.copy(instance_t)
			q_instance[nlabel] = 1  #query value P(Li=1)
			e_instance[nlabel] = -1
		return e, q



	def adjust(self,predict):
		range_o= load_variance()
		for idx,instance in enumerate(predict):
			ones = numpy.count_nonzero(instance)
			q_pr=self.lq_prob[idx]
			while(range_o[0] > ones and not q_pr.empty ):
				jdx = q_pr.get()[2]
				instance[idx][jdx] = 1
				ones = numpy.count_nonzero(instance)
		return predict			

	def classify_batch(self,test):
		predict = reorderingDS(teste)
		#predict = test
		for i in range(0,len(self.nlabels)):
			(ev , query) = generate_EQ(i,predict)
			predict = classify_qev(ev , query , i, self.models[i])
		predict = adjust(predict)	
		predict = undo_OrderingDS(predict)
		return predict	



class MClassifierCC1(MClassifierCCG):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes)

	def preprocess(self):
		logging.info('-- CC1 Preprocess --')
		subsets=[]
		subsets.append(self.train)
		return subsets

	def generate_EQ(self, nlabel,predict):
		q =[]
		e =[]
		l= self.order[nlabel]
		n = self.nlabels 
		unknown_v = []
		for i in range(n):
			unknown_v.append(-1)
		for instance in predict:
			if(nlabel ==0):
				instance[0 : self.nlabels] = unknown_v
			q_instance = instance
			e_instance = instance
			q_instance[l] = 1  #query value P(Li=1)
			e_instance[l] = -1
		return e, q	 

	def classify_batch(self,teste):		
		for i in range(0,len(self.nlabels)):
			(ev , query) = generate_EQ(i,predict)
			predict = classify_qev(ev , query , i, self.models[0])
		predict=adjust(predict)
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
		result=[]
		query=[]
		for instance in test:
			q_instance = numpy.copy(instance)
			unknow_labels = numpy.zeros(self.labels)
			unknow_labels.fill(-2)
			q_instance[0:self.labels] = unknow_labels[ : ]
			query.append(q_instance)
		aspn = self.models[0]
		mpe_values = aspn.mpe(query,self.labels)
		for instance, v_mpe in zip(query,mpe_values):
			instance[:self.labels] = value_mpe[ : ]
			result.append(instance)
		return  result
		

class MClassifierLP(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes):
		 super().__init__(learn_algorithm, train, name, nlabels, nattributes)

	def preprocess(self):
		logging.info('-- LP Preprocess --')
		subsets=[]
		subsets.append(self.train)
		return subsets

	def kbits(self, n, k):
		result = []
		for bits in itertools.combinations(range(n), k):
			s = ['0'] * n
			for bit in bits:
				s[bit] = '1'
			result.append(int(x) for x in list(''.join(s)))
		return result

	def generate_EQ(self, teste, var):
		power_set =[]
		if  var:
			#whole powerset
			for i in var:
				power_set.append(kbits(self.nlabels, i))
		else:
			#powerset, where the number of 1s is in range label cardinality +- variance(var)
			power_set = list(itertools.product([0, 1], repeat = self.nlabels))
			power_set = power_set[1:-1]
		q_v = []
		e_v = []
		unknown_v = []
		for i in range(self.nlabels):
			unknown_v.append(-1)
		for instance in teste:
			q_instance = instance
			e_instance = instance
			for v_labels in power_set:
				q_instance[:self.nlabels] = v_labels
				e_instance[:self.nlabels] = unknown_v 
				q_v.append(q_instance)
				e_v.append(e_instance)
		q_v = numpy.array(q_v).astype(int)
		e_v = numpy.array(e_v).astype(int)
		return q_v , e_v, len(power_set)

	

	def classify_qev(ev, query, a_spn ):
		log_proba = a_spn.query_PC(ev,query)
		return log_proba
	

	def classify_batch(self,teste):
		var = load_variance()
		q , e , lps = generate_EQ(teste, var)
		a_spn = self.models[0]
		ps_predict = classify_qev(e ,q, a_spn )
		maxprob = -999 
		valmax = []
		predict = []
		for idx , instance in enumerate(ps_predict):
			val == query[idx]
			a_prob = float(instance)
			if a_prob > maxprob:
				maxprob = a_prob
				valmax=val
			if((idx % lps) == 0 and not maxprob==-999):
				maxprob = -999
				predict.append(valmax) 	
		return predict 		