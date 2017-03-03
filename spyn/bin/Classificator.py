import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import dataset

import numpy
from numpy.testing import assert_almost_equal

import random

import datetime

import os

import logging

from algo.learnspn import LearnSPN

from spn import NEG_INF
from spn.utils import stats_format

class Classificator(object):
	def __init__(self,
				 dataset,	
				 atributtes,
				 labels)	
		self.dataset = dataset
		self.train = dataset.load_train_csvs(dataset)
		self.atributtes = atributtes
		self.labels = labels
	def train(self)
		self.chosser_spn = ChooserSPN()
		self.spn=self.chooser_spn.train()
		
	def gen_ev_q(self,
				 label,
				 test)
		for instance in test
			q_instance = instance[0:label+1]
			q_instance.append (instance [self.labels+1:])
			qlist.append(q_instance)	
			e_instance = instance[0:label]
			e_instance.append(_1)  #for unknow value
			e_instance.append (instance [self.labels+1:])
			elist.append(e_instance)
		query = numpy.array(qlist).astype(type)
		evidence = numpy.array(elist).astype(type)
		return [query, evidence]
				 	
	def classify(self,
				 label,
				 test)
		 #// P(q | ev) = P(ev,q) / P(e)
		(query,evidence)=gen_ev_q(label,test)
		#compute P(query,evidence)
		ll_q = self.chooser_spn.compute-ll(self.spn,query, '')
		#compute P(e)
		ll_e = self.chooser_spn.compute-ll(self.spn,evidence, '')
				 
