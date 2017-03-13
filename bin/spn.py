import sys
sys.path.append('../')
from spyn.bin.choose_spn import ChooserSPN
from libra import call_idspn
from bin import dataset
import logging

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
		return self.name_spn

	def batch_prob_cond(self,
						evidence,
						query):
		pass

	def batch_mpe(self,
				  evidence,
				  query):
		pass


class SPNAC(object):
	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name = name

	def learn(self):
		logging.info('-- Learn SPN AC --')
		self.chooser_spn = ChooserSPN(self.dataset,'results/models_ac/', self.name)
		return self.chooser_spn.learn_model(True)

	def batch_prob_cond(self,
						evidence,
						query):
		pass

	def batch_mpe(self,
				  evidence,
				  query):
		pass

	

class SPNAL(object):

	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name = name

	def learn(self):
		logging.info('-- Learn SPN AL --')
		self.chooser_spn = ChooserSPN(self.dataset,'results/models_al/', self.name)
		return self.chooser_spn.learn_model(False)

	def batch_prob_cond(self,
						evidence,
						query):
		pass

	def batch_mpe(self,
				  evidence,
				  query):
		pass

