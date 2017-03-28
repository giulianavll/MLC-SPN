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
				 nattributes,
				 bagg,
				 approach):
		self.learn_algorithm = learn_algorithm
		self.train = train
		self.nlabels = nlabels
		self.nattributes = nattributes
		self.name = name
		self.order = []
		self.models = []
		self.bagg = bagg
		self.approach = approach


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
		aspn.learn(self.bagg)
		return aspn

	def generate_SPNAC(self,
					   subset,
					   name):
		aspn = SPNAC(subset, name)
		aspn.learn(self.bagg)	
		return aspn

	def generate_order(self):
		#random order
		order=[i for i in range(self.nlabels)]
		random.shuffle(order)
		self.order = order


	def preprocess(self):
		#implemented in the specific class
		pass

	def create_classifier(self):
		#Create classifier, learning SPNs with the algorithm of the params
		print('----Init classifier construction---')
		self.options = {
		'id' : self.generate_SPNID,
		'al' : self.generate_SPNAL,
		'ac' : self.generate_SPNAC
		}
		subsets = self.preprocess()
		method_learn = self.options[self.learn_algorithm]
		s_sbs = len(subsets)
		for i,s in enumerate(subsets):
			if s_sbs == 1:
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

	def classify_metrics(self, test):
		print('----Init classification---')
		d_predict = self.classify_batch(test)
		print('----Get metrics---')
		self.print_metrics(test,d_predict)
		return 

	def print_metrics(self ,test, predict):
		c_exact = 0
		c_hamming = 0
		c_accuracy = 0
		n = test.shape[0]
		for irs,ios in zip(predict,test):
			r = irs[:self.nlabels]
			o = ios[:self.nlabels]
			and_1 = 0
			or_1 = 0
			bexact = True
			for ir, io in zip(r,o):
				if(ir == io):
					c_hamming += 1
					if(ir == 1):
						and_1 +=  1
						or_1 += 1
				else:
					bexact = False
					or_1 += 1
			if(or_1 != 0):
				c_accuracy +=  and_1 / or_1
			if(bexact == True):
				c_exact += 1
		exact = c_exact / n
		hamming = c_hamming / (n*self.nlabels)
		accuracy = c_accuracy / n
		str_bag=''
		if(not self.bagg):
			str_bag='sb'
		file = open('results/result_classifier/'+self.name+self.learn_algorithm	+self.approach+str_bag, 'w')
		file.write('Exact Match : ' +str(exact))
		file.write("\n")
		file.write('Hamming Score : '+ str(hamming))
		file.write("\n")
		file.write('Accuracy : ' +str(accuracy))
		return


	def classify_batch(self, test):
		pass

	def classify_qev(self,ev, query, l, a_spn ):
		log_prob = a_spn.query_PC(ev,query)
		for i,lp in enumerate(log_prob):
			if lp < log(0.5):
				val = query[i][l]
				if val == 1:
					query[i][l] = 0
					#Store priority queue for find the best probability of label with value 0 must be 1 CCG and PCC
					if self.order:
						self.lq_prob[i].put((-lp,i,l))
				else:
					query[i][l] = 1
		return query


class MClassifierBR(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes,bagg,approach)

	def preprocess(self):
		#BR preprocess Generate l subsets for each label-classifier 
		logging.info('-- BR Preprocess --')
		subsets = []
		dataset = self.train
		n = dataset.shape[0]	
		for l in range (0,self.nlabels):
			train_l = numpy.zeros((n,1+self.nattributes),dtype=numpy.int8)
			strain_l[:,0]=dataset[:,l]
			train_l[:,1:]=dataset[:,self.nlabels:]
			subsets.append(train_l)
		return subsets

	def generate_eq(self,l,test):
		n=test.shape[0]
		q =numpy.zeros((n,1+self.nattributes),dtype=numpy.int8)
		e =numpy.zeros((n,1+self.nattributes),dtype=numpy.int8)
		unknown_v = numpy.full((n, 1), -1, dtype='int8')
		query_v = numpy.full((n, 1), 1, dtype='int8')
		q[:,0] = query_v[:,0]
		e[:,0] = unknow_v[:,0]
		q[:,1:] = test[:,self.nlabels:]
		e[:,1:] = test[:,self.nlabels:]
		return e, q

	def classify_batch(self, test):
		predict =numpy.zeros_like(test)
		t_test= numpy.copy(test)
		for i in range(0,len(self.nlabels)):
			(ev , query) = self.generate_eq(i,t_test)
			predict_i = self.classify_qev(ev , query , 0, self.models[i])
			predict[:,i]=predict_i[:,0]
		return predict	



class MClassifierCCG(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach):
		super().__init__(learn_algorithm, train,name,nlabels,nattributes,bagg,approach)
		


	def reorderingDS(self,dataset):
		perm = numpy.argsort(self.order)
		r_dataset = dataset[:,perm]
		return r_dataset


	def undo_OrderingDS(self,dataset):
		o_dataset = dataset[:,self.order]
		return o_dataset
		

	def preprocess(self):
		logging.info('-- CC Preprocess --')
		self.generate_order()
		subsets = []
		dataset = self.train
		n=dataset.shape[0]
		#Reordering in training set given the label order
		dataset= self.reorderingDS(dataset)
		for l in range (self.nlabels): 
			
			train_l = numpy.zeros((n,l+1+self.nattributes),dtype=numpy.int8)
			train_l[:,:l+1]=dataset[:,:l+1]
			train_l[:,l+1:]=dataset[:,self.nlabels:]
			subsets.append(train_l)
		return subsets

	def generate_eq(self, nlabel,predict):
		n = predict.shape[0]
		unknown_v = numpy.full((n, 1), -1, dtype='int8')
		query_v = numpy.full((n, 1), 1, dtype='int8')
		q= numpy.zeros((n,nlabel+1+self.nattributes),dtype=numpy.int8)
		e= numpy.zeros((n,nlabel+1+self.nattributes),dtype=numpy.int8)

		q[:,:nlabel] = predict[:,:nlabel]
		e[:,:nlabel] = predict[:,:nlabel]
		q[:,nlabel]= query_v[:,0]
		e[:,nlabel]= unknown_v[:,0]
		q[:,nlabel+1:] = predict[:,self.nlabels:]
		e[:,nlabel+1:] = predict[:,self.nlabels:]	
		return e, q

	def adjust(self,predict):
		range_v = self.load_variance()
		for idx,instance in enumerate(predict):
			ones = numpy.count_nonzero(instance[:self.nlabels])
			q_pr = self.lq_prob[idx]
			while(range_v[0] > ones and not q_pr.empty ):
				#Add ones until reach the minimal average range in the relevance vector
				jdx = q_pr.get()[2]
				instance[idx][jdx] = 1
				ones = numpy.count_nonzero(instance[:self.nlabels])
		return predict			

	def classify_batch(self,test):
		predict = self.reorderingDS(test)
		self.lq_prob = [queue.PriorityQueue() for i in range(test.shape[0])]
		#predict = test
		for i in range(0,len(self.nlabels)):
			(ev , query) = self.generate_eq(i,predict)
			predict = self.classify_qev(ev , query , i, self.models[i])
		predict = self.adjust(predict)	
		predict = self.undo_OrderingDS(predict)
		return predict	

class MClassifierPCC(MClassifierCCG):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes,bagg,approach)

	def preprocess(self):
		logging.info('-- PCC Preprocess --')
		self.generate_order()
		
		subsets=[]
		subsets.append(self.train)
		return subsets

	def generate_eq(self, n,label,predict):
		n = predict.shape[0]
		query_v = numpy.full((n, 1), 1, dtype='int8')
		q = numpy.zeros_like(predict)
		e = numpy.zeros_like(predict)

		if n==0:
			unknown_v = numpy.full((n, self.nlabels), -1, dtype='int8')
			q[:,:self.nlabels] = unknown_v[:,:]
			e[:,:self.nlabels] = unknown_v[:,:]
			q[:,self.nlabels:] = predict [:,self.nlabels:]
			e[:,self.nlabels:] = predict [:,self.nlabels:]
		else:
			q[:,:] = predict[:,:]
			e[:,:] = predict[:,:]
		q[:,label]= query_v[:,0]
		return e, q	 

	def classify_batch(self,test):
		self.lq_prob = [queue.PriorityQueue() for i in range(test.shape[0])]
		predict=numpy.copy(test)		
		for i in range(0,self.nlabels):
			order_i=self.order[i]
			(ev , query) = self.generate_eq(i,order_i,predict)
			predict = self.classify_qev(ev , query , order_i, self.models[0])
		predict = self.adjust(predict)
		return predict	


class MClassifierMPE(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach):

		super().__init__(learn_algorithm, train,name,nlabels,nattributes,bagg,approach)

	def preprocess(self):
		logging.info('-- MPE Preprocess --')
		subsets=[]
		subsets.append(self.train)
		return subsets


	def classify_batch(self, test):
		result=[]
		query=[]
		query = numpy.copy(test)
		for q_instance in query:
			unknow_labels = numpy.zeros(self.nlabels)
			unknow_labels.fill(-2)
			q_instance[ :self.nlabels] = unknow_labels[ : ]
		aspn = self.models[0]
		mpe_values = aspn.mpe(query,self.nlabels)
		for instance, v_mpe in zip(query,mpe_values):
			instance[:self.nlabels] = v_mpe[ : ]
			result.append(instance)
		result = numpy.array(result).astype('int8')
		return  result
		

class MClassifierLP(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach):

		super().__init__(learn_algorithm, train, name, nlabels, nattributes,bagg,approach)

	def preprocess(self):
		logging.info('-- LP Preprocess --')
		subsets=[]
		subsets.append(self.train)
		return subsets

	def kbits(self, n, k):
		#Return the list of all binary vectors with size n and k ones
		result = []
		for bits in itertools.combinations(range(n), k):
			s = [0] * n
			for bit in bits:
				s[bit] = 1
			result.append(s)
		return result

	def generate_eq(self, test, var):
		power_set =[]
		if  var:
			#powerset, where the number of 1s is in range label cardinality +- variance(var)
			for i in var:
				#concatenate the lists generated by kbits (return list of all vectors with size nlabels and i ones )
				power_set= power_set + kbits(self.nlabels, i)
		else:
			#whole powerset
			power_set =[list(i) for i in list(itertools.product([0, 1], repeat = self.nlabels))]
			power_set = power_set[1:-1]
		predict = numpy.copy(test)	
		unknown_v = numpy.full( self.nlabels, -1, dtype='int8')
		q_v = []
		e_v = []
		for instance in predict:
			q_instance = numpy.copy(instance)
			e_instance = numpy.copy(instance)
			for v_labels in power_set:
				q_instance[:self.nlabels] = v_labels[:]
				e_instance[:self.nlabels] = unknown_v[:] 
				q_v.append(q_instance)
				e_v.append(e_instance)
		q_v = numpy.array(q_v).astype('int8')
		e_v = numpy.array(e_v).astype('int8')
		return (q_v , e_v, len(power_set))

	

	def classify_qev(ev, query, a_spn ):
		log_proba = a_spn.query_PC(ev,query)
		return log_proba
	

	def classify_batch(self,test):
		var = self.load_variance()
		(q , e , s_lp) = self.generate_eq(test, var)
		a_spn = self.models[0]
		ps_predict = self.classify_qev(e ,q, a_spn )
		max_prob = -999 
		val_max = []
		predict = []
		for idx , p in enumerate(ps_predict):
			a_prob = float(p)
			if a_prob > max_prob:
				max_prob = a_prob
				val_max = q[idx]
			if ((idx+1) % s_lp) == 0:
				predict.append(valmax)
				max_prob = -999
				valmax=[]			
		return predict 		