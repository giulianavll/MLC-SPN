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
		self.io=''


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
		# random.shuffle(order)
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

	def print_metrics(self ,test, predict, suffix=''):
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
		name_g = ''.join([i for i in self.name if not i.isdigit()])										

		if( self.bagg):
			str_bag='wb'	
		file = open('results/result_classifier/'+name_g+'/'+self.name+self.learn_algorithm	+self.approach+str_bag+suffix+self.io, 'w')
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
			val = query[i][l]
			if lp < log(0.5):
				if val == 0:
					query[i][l] = 1
					#Store priority queue for find the best probability of label with value 0 must be 1 CCG and PCC
				elif val==1:
					query[i][l] = 0
					if self.order:
						self.putQueue(i,(-lp,i,l))
						#self.lq_prob[i].put((lp,i,l))
				else:
					print('ohh')
					print(val)
			else:
				if val == 0:
					#Store priority queue for find the best probability of label with value 0 must be 1 CCG and PCC
					if self.order:
						self.putQueue(i,(-lp,i,l))
						#self.lq_prob[i].put((lp,i,l))
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

	def getQueue(self,idx):
		return self.lq_prob[idx]

	def putQueue(self,pos,t):
		self.lq_prob[pos].put(t)

	def adjust(self,predict):
		range_v = self.load_variance()
		for idx,instance in enumerate(predict):
			ones = numpy.count_nonzero(instance[:self.nlabels])
			q_pr = self.getQueue(idx)
			#q_pr = self.lq_prob[idx]
			while(range_v[0] > ones and not q_pr.empty ):
				#Add ones until reach the minimal average range in the relevance vector
				jdx = q_pr.get()[2]
				instance[idx][jdx] = 1
				ones = numpy.count_nonzero(instance[:self.nlabels])
		return predict			

	def classify_batch(self,test):
		predict = numpy.copy(test)
		predict[:,:self.nlabels] = self.reorderingDS(test[:,:self.nlabels])
		self.lq_prob = [queue.PriorityQueue() for i in range(test.shape[0])]
		#predict = test
		for i in range(0,self.nlabels):
			(ev , query) = self.generate_eq(i,predict)
			predict = self.classify_qev(ev , query , i, self.models[i])
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
				power_set= power_set + self.kbits(self.nlabels, i)
		else:
			#whole powerset
			power_set =[list(i) for i in list(itertools.product([0, 1], repeat = self.nlabels))]
			power_set = power_set[1:-1]
		predict = numpy.copy(test)	
		unknown_v = numpy.full( self.nlabels, -1, dtype='int8')
		q_v = []
		e_v = []
		for instance in predict:
			for v_labels in power_set:
				q_instance = numpy.copy(instance)
				e_instance = numpy.copy(instance)
				q_instance[:self.nlabels] = v_labels[:]
				e_instance[:self.nlabels] = unknown_v[:] 
				q_v.append(q_instance)
				e_v.append(e_instance)
		q_v = numpy.array(q_v).astype('int8')
		e_v = numpy.array(e_v).astype('int8')
		return (q_v , e_v, len(power_set))

	

	def classify_qev(self,ev, query, a_spn ):
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
				predict.append(val_max)
				max_prob = -999
				val_max=[]			
		return predict 		

class MClassifierSC(MClassifierCCG):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach,io):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes,bagg,approach)
		 self.io=io

	def preprocess(self):
		logging.info('-- PCC Preprocess --')
		self.generate_order()
		subsets=[]
		subsets.append(self.train)
		return subsets

	def generate_eq(self, nl,label,predict):
		n = predict.shape[0]
		query_v = numpy.full((n, 1), 1, dtype='int8')
		q = numpy.zeros_like(predict)
		e = numpy.zeros_like(predict)
		if nl==0:
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
		# predict = self.adjust(predict)
		return predict	






class MClassifierSCIO(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach,io):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes,bagg,approach)
		 self.io = io

	def preprocess(self):
		logging.info('--  Preprocess --')
		self.order=[]
		subsets=[]
		subsets.append(self.train)
		return subsets

	def getQueue(self,idx):
		return self.lq_prob

	def putQueue(self,pos,t):
		self.lq_prob.put(t)

	def getOrder_S(self,example):
		a_spn=self.models[0]
		log_probs=[]
		unknown_v = numpy.full(self.nlabels, -1, dtype='int8')
		qe= numpy.copy(example)
		for i in range(0,self.nlabels):
			ev = numpy.copy(qe)
			query = numpy.copy(qe)
			query[i] = 0
			log_probs.append(a_spn.query_PC(ev,query)[0])
		order = numpy.argsort(log_probs)	
		return order
	
	def getOrder_D(self,example):
		a_spn=self.models[0]
		order=[]
		unknown_v = numpy.full(self.nlabels, -1, dtype='int8')
		qe= numpy.copy(example)
		qe[:self.nlabels]=unknown_v
		for i in range(0,self.nlabels):
			log_probs=[]
			for j in range(0,self.nlabels):
				if not (j in order):
					ev = numpy.copy(qe)
					query = numpy.copy(qe)
					query[j] = 0
					log_probs.append(a_spn.query_PC(ev,query)[0])
				else:
					log_probs.append(99999)
			order_probs=numpy.argsort(log_probs)
			idx_h = order_probs[0]
			order.append(idx_h)
			value = log_probs[idx_h]
			if value > log(0.5):
				qe[idx_h] = 0
			else:
				qe[idx_h] = 1
		return order

	def adjust(self,instance):
		range_v = self.load_variance()
		ones = numpy.count_nonzero(instance[:self.nlabels])
		q_pr = self.getQueue(0)
		#q_pr = self.lq_prob[idx]
		while(range_v[0] > ones and not q_pr.empty ):
			#Add ones until reach the minimal average range in the relevance vector
			jdx = q_pr.get()[2]
			instance[jdx] = 1
			ones = numpy.count_nonzero(instance[:self.nlabels])
		return instance		

	def generate_eq(self, nl,label,predict):
		q = numpy.zeros_like(predict)
		e = numpy.zeros_like(predict)
		if nl==0:
			unknown_v = numpy.full( self.nlabels, -1, dtype='int8')
			q[:self.nlabels] = unknown_v[:]
			e[:self.nlabels] = unknown_v[:]
			q[self.nlabels:] = predict [self.nlabels:]
			e[self.nlabels:] = predict [self.nlabels:]
		else:
			q[:] = predict[:]
			e[:] = predict[:]
		q[label]= 1
		return e, q	 
	

	def classify_qev(self,ev, query, l, a_spn ):
		log_prob = a_spn.query_PC(ev,query)
		lp=log_prob[0]
		val=query[l]	
		if lp < log(0.5):
			if val == 0:
				query[l] = 1
				#Store priority queue for find the best probability of label with value 0 must be 1 CCG and PCC
			elif val==1:
				query[l] = 0
				self.putQueue(0,(-lp,0,l))
			else:
				print('ohh')
				print(val)
		else:
			if val == 0:
				#Store priority queue for find the best probability of label with value 0 must be 1 CCG and PCC
				self.putQueue(0,(-lp,0,l))
		return query
		

	def classify_batch(self,test):
		predict=numpy.copy(test)
		for x,example in enumerate(test):
			self.lq_prob =queue.PriorityQueue()
			if self.io == 's':
				self.order=self.getOrder_S(example)
			elif self.io =='d':
				self.order=self.getOrder_D(example)
			predict_e=numpy.copy(example)

			for i in range(0,self.nlabels):
				order_i=self.order[i]
				(ev , query) = self.generate_eq(i,order_i,predict_e)
				predict_e = self.classify_qev(ev , query , order_i, self.models[0])
			predict_e = self.adjust(predict_e)
			predict[x]=predict_e
		return predict	

class MClassifierLPD(Classifier):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach):
		super().__init__(learn_algorithm, train, name, nlabels, nattributes,bagg,approach)

	def preprocess(self):
		logging.info('--  Preprocess --')
		self.order=[]
		subsets=[]
		subsets.append(self.train)
		return subsets

	def get_jprob(self, query):
		spn= self.models[0]
		return spn.query(query)

	def get_lbs(self, label_set,pls, new_label_set,new_pls):
		for (idx,npl) in enumerate(new_pls):
			if npl > pls[idx]:
				pls[idx]=npl
				label_set[idx]= new_label_set[idx]
		return label_set,pls

	def kbits(self, n, k):
		#Return the list of all binary vectors with size n and k ones
		result = []
		for bits in itertools.combinations(range(n), k):
			s = [0] * n
			for bit in bits:
				s[bit] = 1
			result.append(s)
		return result


	def get_predictionlpd(self, label_set):
		pls= self.get_jprob(label_set)
		for l in range(0, self.nlabels):
			new_label_set = numpy.copy(label_set)
			for idx,lc in enumerate(label_set):
				if label_set[idx][l]==1:
					new_label_set[idx][l]=0
				else:
					new_label_set[idx][l]=1
			new_pls = self.get_jprob(new_label_set)
			(label_set,pls) = self.get_lbs(label_set,pls, new_label_set,new_pls)
		imax=numpy.argmax(pls)
		return label_set[imax]	



	def get_predictionlp(self, queries):
		ev= numpy.copy(queries)
		unknown_v = numpy.full((ev.shape[0],self.nlabels), -1, dtype='int8')
		ev[:,:self.nlabels]=unknown_v
		a_spn = self.models[0]
		pls=a_spn.query_PC(ev,queries)
		pmax=-9999
		preq=numpy.zeros_like(queries[0])
		for q,p in zip(queries,pls):

			if p>pmax:
				pmax=p
				preq=q
		return preq


	def load_vlabel(self):
		vlabels = self.train[:,0:self.nlabels]
		dvlabels = numpy.vstack({tuple(row) for row in vlabels})
		return dvlabels

	def load_vlabelp(self):
		var = self.load_variance()
		power_set=[]
		if  var:
			#powerset, where the number of 1s is in range label cardinality +- variance(var)
			for i in var:
				#concatenate the lists generated by kbits (return list of all vectors with size nlabels and i ones )
				power_set= power_set + self.kbits(self.nlabels, i)
		else:
			power_set =[list(i) for i in list(itertools.product([0, 1], repeat = self.nlabels))]
			power_set = power_set[1:-1]
		return numpy.asarray(power_set)

	def concate_LE(self,example, label_set):
		lp = numpy.full((label_set.shape[0],example.shape[0]),example, dtype=numpy.int8)
		for idx,labelv in enumerate(label_set):
			lp[idx,:self.nlabels]=labelv
		return lp
		
	def classify_batch(self,test):
		predict = numpy.copy(test)
		if self.approach =='ec':
			label_set = self.load_vlabel()
		elif self.approach =='e2':
			label_set = self.load_vlabelp()
		for x,example in enumerate(test):
			label_q = self.concate_LE(example,label_set)
			if self.approach =='ec':
				predict[x]= self.get_predictionlpd(label_q)
			elif self.approach =='e2':
				predict[x]= self.get_predictionlp(label_q)
			
		return predict




class MClassifierPC(MClassifierSCIO,MClassifierLPD):
	def __init__(self,learn_algorithm, train,name,nlabels,nattributes,bagg,approach,psm,npool=10):
		 super().__init__(learn_algorithm, train,name,nlabels,nattributes,bagg,approach,'')
		 self.psm = psm
		 self.weights_avg = []
		 self.weights_vot = []
		 self.predict_v = []
		 # print(npool)
		 self.npool = npool

	def generate_order(self):
		for i in range(self.npool):
			self.order.append(numpy.random.permutation(self.nlabels) )


	def preprocess(self):
		logging.info('-- PS Preprocess --')
		if self.approach == 'psc':
			self.generate_order()

		elif self.approach == 'pec':
			self.label_set = self.load_vlabel()
		subsets=[]
		subsets.append(self.train)
		return subsets

    
	def classify_qev(self,ev, query, a_spn):
		log_prob = a_spn.query_PC(ev,query)
		lp=log_prob[0]	
		return lp 

	
	def get_Weights_sq(self,example):
		predict_e = numpy.copy(example)
		weights_avg = numpy.zeros((self.nlabels,), dtype=numpy.int8)
		weights_vot = numpy.zeros((self.nlabels,), dtype=numpy.int8)
		
		predict_v = numpy.full((self.npool,predict_e.shape[0]),predict_e,dtype=numpy.int8)
		for l in range(self.nlabels):
			for i,order in enumerate(self.order):
				order_l = order[l]
				(ev , query) = self.generate_eq(l,order_l,predict_v[i])
				log_prob = self.classify_qev(ev , query,self.models[0])
				weights_avg[order_l] = weights_avg[order_l] + log_prob
				if log_prob >= log(0.5):
					weights_vot[order_l] = weights_vot[order_l] + 1
					query[order_l]=1
				else:
					query[order_l]=0
				predict_v[i] = query
		self.weights_avg = weights_avg/self.npool
		self.weights_vot = weights_vot/self.npool
		self.predict_v = predict_v
		
		return 


	def get_Weights_lp(self,example):
		predict_e = numpy.copy(example)
		weights_vot = numpy.zeros((self.nlabels,), dtype=numpy.int8)
		weights_avg= numpy.full(self.nlabels,0.0)
		weights_avg_a = numpy.full(self.nlabels,0.0)
		predict_v = numpy.full((self.npool,predict_e.shape[0]),predict_e,dtype=numpy.int8)
		queries = self.concate_LE(example,self.label_set)
		pq= self.get_jprob(queries)
		for l in range(0, self.nlabels):
			nqueries = numpy.copy(queries)
			for idx,lc in enumerate(queries):
				if queries[idx][l]==1:
					nqueries[idx][l]=0
				else:
					nqueries[idx][l]=1
			npq = self.get_jprob(nqueries)
			(queries,pq) = self.get_lbs(queries,pq, nqueries,npq)
		ind = numpy.argpartition(pq, -self.npool)[-self.npool:]	
		acp =numpy.sum(pq)	
		for i,ix in enumerate(ind):
			q=queries[ix]
			predict_v[i]=q
			p = numpy.exp(pq[ix])
			for lx,v in enumerate(q[0:self.nlabels]):
				if v == 1:
					weights_vot[lx]	 = weights_vot[lx] + 1
					weights_avg[lx] = weights_avg[lx] + p
			
				else :
					weights_avg_a[lx] = weights_avg[lx] + p			
		self.weights_avg = weights_avg/self.npool	
		if(self.approach =='pec'):
			self.weights_avg_a = weights_avg_a/self.npool
		self.weights_vot = weights_vot/self.npool
		self.predict_v = predict_v
		return 

	def choose_Avg(self,example) :
		predict_e = numpy.copy(example)
		self.lq_prob =queue.PriorityQueue()
		if len(self.weights_avg)==0:
			if self.approach == 'psc':
				self.get_Weights_sq(example)
			elif self.approach == 'pec':
				self.get_Weights_lp(example)
		if self.approach == 'psc':		
			for i,w in enumerate(self.weights_avg):
				if w >= log(0.5):
					predict_e[i]= 1
				else:
					predict_e[i]= 0
					self.putQueue(0,(-w,0,i))
		elif self.approach == 'pec':
			# print(self.weights_avg_a)
			# print(self.weights_avg)
			for i,wp in enumerate(self.weights_avg):
				wa = self.weights_avg_a[i]
				if wp >= wa:
					predict_e[i]= 1
				else:
					predict_e[i]= 0
					self.putQueue(0,(-wp+wa,0,i))
		# print('predict')
		# print(example)
		# print(predict_e)
		return self.adjust(predict_e)
		

	def choose_Vot(self,example):
		predict_e = numpy.copy(example)
		self.lq_prob =queue.PriorityQueue()
		if len(self.weights_vot)==0:
			if self.approach == 'psc':
				self.get_Weights_sq(example)	
			elif self.approach == 'pec':
				self.get_Weights_lp(example)
		for i,w in enumerate(self.weights_vot):
			if w >= 0.5:
				predict_e[i]= 1
			else:
				predict_e[i]= 0
				self.putQueue(0,(-w,0,i))
		return self.adjust(predict_e)
	
	def choose_Max(self,example):
		predict_e = numpy.zeros_like(example)
		# print('---example---')
		self.lq_prob =queue.PriorityQueue()
		if len(self.predict_v)==0:
			if self.approach == 'psc':
				self.get_Weights_sq(example)
			elif self.approach == 'pec':
				self.get_Weights_lp(example)
		log_max = -99999
		probs={}
		for i in range(len(self.order)):
			pi_str = str(self.predict_v[i,:self.nlabels])
			if not pi_str in probs:
				q=numpy.copy(self.predict_v[i])
				log_prob = self.models[0].query(q)[0]
				probs[pi_str]=log_prob
				if log_max<log_prob:
					log_max=log_prob
					predict_e=q
		return self.adjust(predict_e)
	


	def classify_batch(self,test):
		predict_v=[]
		predict0 = numpy.zeros_like(test)

		if self.psm=='all':
			predict1 = numpy.zeros_like(test)
			predict2 = numpy.zeros_like(test)
		else:
			predict1 = []
			predict2 = []
		for x,example in enumerate(test):	
			predict_e=numpy.copy(example)
			self.weights_avg = []
			self.weights_vot = []
			self.predict_v = []
			if (self.psm =='vot'):
				predict_e = self.choose_Vot(example)				
				predict0[x]=predict_e

			elif (self.psm =='avg'):
				predict_e = self.choose_Avg(example)
				predict0[x]=predict_e

			elif (self.psm =='max'):
				predict_e = self.choose_Max(example)	
				predict0[x]=predict_e

			elif (self.psm == 'all'):
				predict_e0 = self.choose_Vot(example)
				predict0[x]=predict_e0

				predict_e1 = self.choose_Avg(example)	
				predict1[x]=predict_e1

				predict_e2 = self.choose_Max(example)	
				predict2[x]=predict_e2

		predict_v.append(predict0)
		if (len(predict1)>0 and len(predict2)>0):
			predict_v.append(predict1)
			predict_v.append(predict2)
		return predict_v	

	def classify_metrics(self, test):
		print('----Init classification---')
		d_predict = self.classify_batch(test)
		print('----Get metrics---')
		names=['vot','avg','max']
		for i,p in enumerate(d_predict):
			if self.psm == 'all':
				self.print_metrics(test,p, names[i])
			else:
				self.print_metrics(test,p, self.psm)
		return 
