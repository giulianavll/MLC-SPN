import sys
sys.path.append('../')
from spyn.bin.choose_spn import ChooserSPN
from libra import call_idspn
from bin import dataset
import logging
import os


class SPNID(object):
	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name = name

	def learn(self):
		logging.info('-- Learn SPN ID --')
		#check if exists a learned model
		# path = 'results/models_l/'
		# files = [i for i in os.listdir(path) if i.startswith(self.name)]
		# if files:
		# 	self.name_spn = files[0].partition('.')[0]
		# else:
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

	def load_args(self, path):
		args = []
		if os.path.isfile(path):
			file = open(path, 'r')
			args = file.readline().split(',')
			args = list(map(float, args))
		return args	

	def learn(self,bagg):
		logging.info('-- Learn SPN AL --')
		path = 'results/models_al/'+self.name+'/args'
		args = self.load_args(path)
		d_size = self.dataset.shape[0]
		comp = 1
		if d_size<600:
			comp=10
		elif d_size<2500:
			comp=4
		else:
			comp=2
		self.chooser_spn = ChooserSPN(self.dataset,'results/models_al/', self.name)
		(self.spn,g_f,min_i,alpha) = self.chooser_spn.learn_model(False,args,comp,bagg)
		if not args:
			self.save_args(path,[g_f,min_i,alpha])
		return 

	def save_args(self,path,args):
		file = open(path, 'w')
		file.write(str(args[0])+','+str(args[1])+','+str(args[2]))
	    

		
	def query_PC(self, evidence, query):
		#// P(q | ev) = P(ev,q) / P(e)  --> log(P(q)) - log(P(e))
		lle = self.chooser_spn.compute_ll(self.spn,evidence)
		llq = self.chooser_spn.compute_ll(self.spn,query)
		pc =[]
		for pq , pe in zip(llp,lle):
			pc.append(pq - pe)
		return pc 


	def mpe(self, query, n):
		result_mpe=self.chooser_spn.val_mpe(self.spn , query,n)
		return result_mpe


class SPNAC(SPNAL):
	def __init__(self,
		         dataset,
		         name):
		self.dataset = dataset
		self.name = name

	def learn(self,bagg):
		logging.info('-- Learn SPN AC --')
		path = 'results/models_al/'+self.name+'/args'
		args=self.load_args(path)
		comp = 1
		d_size = self.dataset.shape[0]
		if d_size<600:
			comp=10
		elif d_size<2500:
			comp=4
		else:
			comp=2
		self.chooser_spn = ChooserSPN(self.dataset,'results/models_ac/', self.name)
		(self.spn,g_f,min_i,alpha) = self.chooser_spn.learn_model(True,args,comp,bagg)
		if not args:
			self.save_args(path,[g_f,min_i,alpha])
		return 
	