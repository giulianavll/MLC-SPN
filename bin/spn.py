from spyn.bin import choose_spn
from libra import call_idspn
from bin import dataset

class SPNID(SPN):
	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name=name

	def learn(self):
		dataset.numpy_2_file(self.dataset, self.name)
		call_idspn.learn_Libra(self.name)
		return self.name

	def batch_prob_cond(self,
						evidence,
						query):
		pass

	def batch_mpe(self,
				  evidence
				  query):
		pass


class SPNAC(SPN):
	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name = name

	def learn(self):
		self.chooser_spn = ChooserSPN(self.dataset,'results/models_ac/', self.name)
		return self.chooser_spn.learn_model(True)

	def batch_prob_cond(self,
						evidence,
						query):
		pass

	def batch_mpe(self,
				  evidence
				  query):
		pass

	

class SPNAL(SPN):

	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name=name

	def learn(self):
		
		self.chooser_spn = ChooserSPN(self.dataset,'results/models_al/', self.name)
		return self.chooser_spn.learn_model(False)

	def batch_prob_cond(self,
						evidence,
						query):
		pass

	def batch_mpe(self,
				  evidence
				  query):
		pass

