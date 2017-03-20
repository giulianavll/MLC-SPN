import sys
sys.path.append('../')
from spyn.bin.choose_spn import ChooserSPN
from libra import call_idspn
from bin import dataset
import logging

#a_spn.query_PC(ev,query)
#aspn.mpe(query,self.labels)


class SPNID(object):
	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name = name

	def learn(self):
		logging.info('-- Learn SPN ID --')
		dataset.numpy_2_file(self.dataset, self.name)
		self.name_spn = call_idspn.create_IDnetworks(self.name)
		return 

	def query_PC(self, evidence, query):
		dataset.numpy_2_file(evidence, self.name+'.ev')
		dataset.numpy_2_file(query, self.name+'.q')
		probs = call_idspn.inference_mg(self.name, self.name_spn)

	def mpe(self, query, n):
		v_mpe =[]
		return v_mpe

class SPNAL(object):

	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name = name

	def learn(self):
		logging.info('-- Learn SPN AL --')
		self.chooser_spn = ChooserSPN(self.dataset,'results/models_al/', self.name)
		self.spn = self.chooser_spn.learn_model(False)
		return 


	def query_PC(self, evidence, query):
		#// P(q | ev) = P(ev,q) / P(e)  --> P(q) - P(e)
		lle = self.chooser_spn.compute_ll(self.spn,evidence)
		llq = self.chooser_spn.compute_ll(self.spn,query)
		pc =[]
		for pq , pe in zip(llp,lle):
			pc.append(pq - pe)
		return pc 


	def mpe(self, query, n):
		self.chooser_spn.val_mpe(self.spn , query,n)


class SPNAC(SPNAL):
	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name = name

	def learn(self):
		logging.info('-- Learn SPN AC --')
		self.chooser_spn = ChooserSPN(self.dataset,'results/models_ac/', self.name)
		self.spn = self.chooser_spn.learn_model(True)
		return 
	

