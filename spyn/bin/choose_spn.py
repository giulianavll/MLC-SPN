import argparse
import sys
sys.path.append('spyn/')
try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import dataset
import collections
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
      def __init__(self,
                   dataset,
                   path,
                   name):

          self.dataset=name
          self.train=dataset
          self.out_path = path+name+'/'
          os.makedirs(self.out_path, exist_ok=True)

          
      def learn_model(self,cltree_leaves,args,comp,bgg):
          #set parameters for learning AC (cltree_leaves=True)and AL(cltree_leaves=false) 
          print('-------MODELS CONSTRUCTION-----------')
          verbose = 1
          n_row_clusters = 2
          cluster_method = 'GMM'
          seed = 1337
          n_iters = 100
          n_restarts = 4
          cluster_penalties = [1.0]
          sklearn_Args = None
          if not args:
            g_factors = [5,10,15]  
            min_inst_slices = [10,50,100] 
            alphas = [0.1, 0.5 ,1.0,2.0 ]
          else:
            g_factors = [args[0]]  
            min_inst_slices = [args[1]]
            alphas = [args[2]]
          # setting verbosity level
          if verbose == 1:
              logging.basicConfig(level=logging.INFO)
          elif verbose == 2:
              logging.basicConfig(level=logging.DEBUG)

          # logging.info("Starting with arguments:\n")
         
          if sklearn_Args is not None:
              sklearn_key_value_pairs = sklearn_translate(
                  {ord('['): '', ord(']'): ''}).split(',')
              sklearn_args = {key.strip(): value.strip() for key, value in
                              [pair.strip().split('=')
                               for pair in sklearn_key_value_pairs]}
          else:
              sklearn_args = {}
          # logging.info(sklearn_args)

          # initing the random generators
          MAX_RAND_SEED = 99999999  # sys.maxsize
          rand_gen = random.Random(seed)
          numpy_rand_gen = numpy.random.RandomState(seed)

          #
          # elaborating the dataset
          #
          
          dataset_name = self.dataset
          # logging.info('Loading datasets: %s', dataset_name)
          train=self.train
          n_instances = train.shape[0]
          
          #
          # estimating the frequencies for the features
          # logging.info('')
          freqs, features = dataset.data_2_freqs(train)
          best_train_avg_ll = NEG_INF
          best_state = {}
          best_test_lls = None
          index = 0
          spns = []
          for g_factor in g_factors:
              for cluster_penalty in cluster_penalties:
                  for min_inst_slice in min_inst_slices:
                      print('model')
                      # Creating the structure learner
                      learner = LearnSPN(g_factor=g_factor,
                                        min_instances_slice=min_inst_slice,
                                        # alpha=alpha,
                                        row_cluster_method=cluster_method,
                                        cluster_penalty=cluster_penalty,
                                        n_cluster_splits=n_row_clusters,
                                        n_iters=n_iters,
                                        n_restarts=n_restarts,
                                        sklearn_args=sklearn_args,
                                        cltree_leaves=cltree_leaves,
                                        rand_gen=numpy_rand_gen)

                      learn_start_t = perf_counter()
                          
                      # build an spn on the training set
                      if(bgg):
                        spn = learner.fit_structure_bagging(data=train,
                                                           feature_sizes=features,
                                                           n_components=comp)
                      else:
                        spn = learner.fit_structure(data=train,
                                                  feature_sizes=features)
                      

                      learn_end_t = perf_counter()
                      n_edges = spn.n_edges()
                      n_levels = spn.n_layers()
                      n_weights = spn.n_weights()
                      n_leaves = spn.n_leaves()

                      #
                      # smoothing can be done after the spn has been built
                      for alpha in alphas:
                          # logging.info('Smoothing leaves with alpha = %f', alpha)
                          spn.smooth_leaves(alpha)
                          spns.append(spn)

                          # Compute LL on training set
                          # logging.info('Evaluating on training set')
                          train_ll = 0.0
                          for instance in train:
                              (pred_ll, ) = spn.eval(instance)
                              train_ll += pred_ll
                          train_avg_ll = train_ll / train.shape[0]

                          # updating best stats according to train ll
                          if train_avg_ll > best_train_avg_ll:
                              best_train_avg_ll = train_avg_ll
                              best_state['alpha'] = alpha
                              best_state['min_inst_slice'] = min_inst_slice
                              best_state['g_factor'] = g_factor
                              best_state['cluster_penalty'] = cluster_penalty
                              best_state['train_ll'] = train_avg_ll
                              best_state['index']= index
                              best_state['name'] =self.dataset

                          # writing to file a line for the grid
                          # stats = stats_format([g_factor,
                          #                       cluster_penalty,
                          #                       min_inst_slice,
                          #                       alpha,
                          #                       n_edges, n_levels,
                          #                       n_weights, n_leaves,
                          #                       train_avg_ll],
                          #                      '\t',
                          #                      digits=5)
                          # index = index + 1

          best_spn = spns[best_state['index']]
          # logging.info('Grid search ended.')
          # logging.info('Best params:\n\t%s', best_state)

          return best_spn, best_state['g_factor'], best_state['min_inst_slice'],best_state['alpha']

      def compute_ll(self,
                   spn,
                   test,
                   name=None):

          # Compute LL on test set
          test_lls = numpy.zeros(test.shape[0])
          # logging.info('Evaluating on test set')
         
          for i, instance in enumerate(test):
            (pred_ll,) = spn.eval(instance)
            test_lls[i] = pred_ll

          if name is not None:
            test_lls_path = self.out_path + '/'+name+'test.lls'
            numpy.savetxt(test_lls_path, test_lls, delimiter='\n')
          return test_lls

      def val_mpe(self,
                  spn,
                  test,
                  n_labels,
                  name=None):

          arg_mpe = {}
          args_mpes=[]
          for  instance in test:
            (arg_mpe, ) = spn.single_mpe(instance)
            mpe= numpy.zeros(n_labels)
            for k, v in arg_mpe.items():
              mpe[k] =v
            args_mpes.append(mpe)
          return args_mpes



