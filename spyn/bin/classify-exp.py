import argparse

import dataset

import numpy
from numpy.testing import assert_almost_equal


import os

import logging

#########################################
# creating the opt parser
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. nltcs)')

parser.add_argument('-k', '--n-row-clusters', type=int, nargs='?',
                    default=2,
                    help='Number of clusters to split rows into' +
                    ' (for DPGMM it is the max num of clusters)')

parser.add_argument('-c', '--cluster-method', type=str, nargs='?',
                    default='GMM',
                    help='Cluster method to apply on rows' +
                    ' ["GMM"|"DPGMM"|"HOEM"]')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')


parser.add_argument('-g', '--g-factor', type=float, nargs='+',
                    default=[1.0],
                    help='The "p-value like" for G-Test on columns')

parser.add_argument('-i', '--n-iters', type=int, nargs='?',
                    default=100,
                    help='Number of iterates for the row clustering algo')

parser.add_argument('-r', '--n-restarts', type=int, nargs='?',
                    default=4,
                    help='Number of restarts for the row clustering algo' +
                    ' (only for GMM)')

parser.add_argument('-p', '--cluster-penalty', type=float, nargs='+',
                    default=[1.0],
                    help='Penalty for the cluster number' +
                    ' (i.e. alpha in DPGMM and rho in HOEM, not used in GMM)')

parser.add_argument('-s', '--sklearn-args', type=str, nargs='?',
                    help='Additional sklearn parameters in the for of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('-m', '--min-inst-slice', type=int, nargs='+',
                    default=[50],
                    help='Min number of instances in a slice to split by cols')

parser.add_argument('-a', '--alpha', type=float, nargs='+',
                    default=[0.1],
                    help='Smoothing factor for leaf probability estimation')

parser.add_argument('--clt-leaves', action='store_true',
                    help='Whether to use Chow-Liu trees as leaves')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')
#
# parsing the args
args = parser.parse_args()

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)
# I shall print here all the stats

#
# gathering parameters
alphas = args.alpha
min_inst_slices = args.min_inst_slice
g_factors = args.g_factor
cluster_penalties = args.cluster_penalty

cltree_leaves = args.clt_leaves

sklearn_args = None
if args.sklearn_args is not None:
    sklearn_key_value_pairs = args.sklearn_args.translate(
        {ord('['): '', ord(']'): ''}).split(',')
    sklearn_args = {key.strip(): value.strip() for key, value in
                    [pair.strip().split('=')
                     for pair in sklearn_key_value_pairs]}
else:
    sklearn_args = {}
logging.info(sklearn_args)

# initing the random generators
seed = args.seed
