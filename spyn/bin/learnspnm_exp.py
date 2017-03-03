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

class ChooserSPN(object):
	def _init_(self)


	def train-spn(self)
		MAX_RAND_SEED = 99999999  # sys.maxsize
		rand_gen = random.Random(seed)
		numpy_rand_gen = numpy.random.RandomState(seed)

		#
		# elaborating the dataset
		#
		logging.info('Loading datasets: %s', args.dataset)
		(dataset_name,) = args.dataset
		train, test = dataset.load_train_test_csvs(dataset_name)

		n_instances = train.shape[0]
		n_test_instances = test.shape[0]
		#
		# estimating the frequencies for the features
		logging.info('Estimating features on training set...')
		freqs, features = dataset.data_2_freqs(train)
		
		
		logging.info('Opening log file...')
		date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		out_path = args.output + dataset_name + '_' + date_string
		out_log_path = out_path + '/exp.log'
		#
		# creating dir if non-existant
		if not os.path.exists(os.path.dirname(out_log_path)):
			os.makedirs(os.path.dirname(out_log_path))
	
		best_train_avg_ll = NEG_INF
		index = 0
		best_state = {}
		best_test_lls = None
		spn_vector=[]
		preamble = ("""g-factor:\tclu-pen:\tmin-ins:\talpha:\tn_edges:""" +
					"""\tn_levels:\tn_weights:\tn_leaves:""" +
					"""\ttrain_ll\n""")

		with open(out_log_path, 'w') as out_log:

			out_log.write("parameters:\n{0}\n\n".format(args))
			out_log.write(preamble)
			out_log.flush()
			#
			# looping over all parameters combinations
			for g_factor in g_factors:
				for cluster_penalty in cluster_penalties:
					for min_inst_slice in min_inst_slices:

						#
						# Creating the structure learner
						learner = LearnSPN(g_factor=g_factor,
										   min_instances_slice=min_inst_slice,
										   # alpha=alpha,
										   row_cluster_method=args.cluster_method,
										   cluster_penalty=cluster_penalty,
										   n_cluster_splits=args.n_row_clusters,
										   n_iters=args.n_iters,
										   n_restarts=args.n_restarts,
										   sklearn_args=sklearn_args,
										   cltree_leaves=cltree_leaves,
										   rand_gen=numpy_rand_gen)

						learn_start_t = perf_counter()

						#
						# build an spn on the training set
						#spn = learner.fit_structure(data=train,
						#							feature_sizes=features)
						 spn = learner.fit_structure_bagging(data=train,
						                                     feature_sizes=features,
						                                     n_components=10)

						learn_end_t = perf_counter()
						print('Structure learned in', learn_end_t - learn_start_t,
							  'secs')

						#
						# print(spn)

						#
						# gathering statistics
						n_edges = spn.n_edges()
						n_levels = spn.n_layers()
						n_weights = spn.n_weights()
						n_leaves = spn.n_leaves()

						#
						# smoothing can be done after the spn has been built
						for alpha in alphas:
							logging.info('Smoothing leaves with alpha = %f', alpha)
							spn.smooth_leaves(alpha)
							spn_vector.append(spn)
							#
							# Compute LL on training set
							logging.info('Evaluating on training set')
							train_ll = 0.0
							for instance in train:
								(pred_ll, ) = spn.eval(instance)
								train_ll += pred_ll
							train_avg_ll = train_ll / train.shape[0]
							

							#
							# updating best stats according to training ll
							if train_avg_ll > best_train_avg_ll:
							    best_train_avg_ll = train_avg_ll
							    best_state['alpha'] = alpha
							    best_state['min-inst-slice'] = min_inst_slice
							    best_state['g-factor'] = g_factor
							    best_state['cluster-penalty'] = cluster_penalty
							    best_state['train_ll'] = train_avg_ll
							    best_state['index'] = index

							index = index + 1
							#
							# writing to file a line for the grid
							stats = stats_format([g_factor,
												  cluster_penalty,
												  min_inst_slice,
												  alpha,
												  n_edges, n_levels,
												  n_weights, n_leaves,
												  train_avg_ll,
												  valid_avg_ll,
												  test_avg_ll],
												 '\t',
												 digits=5)
							out_log.write(stats + '\n')
							out_log.flush()

			#
			# writing as last line the best params
			out_log.write("{0}".format(best_state))
			out_log.flush()

			#
			

		
		logging.info('Grid search ended.')
		logging.info('Best params:\n\t%s', best_state)
		best_spn= spn_vector[best_state['index']]
		return best_spn
		
	def compute-ll(test,spn,name)
		#
		# Opening the file for test prediction
		#
		date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		out_path = self.output + name + '_' + date_string
		test_lls_path = out_path + '/test.lls'

		
		# Compute LL on test set
		test_lls = numpy.zeros(test.shape[0])
		logging.info('Evaluating on query set')
		test_ll = 0.0
		for i, instance in enumerate(test):
			(pred_ll, ) = spn.eval(instance)
			test_ll += pred_ll
			test_lls[i] = pred_ll
		test_avg_ll = test_ll / test.shape[0]
		# saving the best test_lls
		numpy.savetxt(test_lls_path, test_lls, delimiter='\n')
		return test_lls
