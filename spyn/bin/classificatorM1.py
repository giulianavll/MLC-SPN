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

class ClassificatorM1(Classificator):
	def __init__(self,
				 dataset,	
				 atributtes,
				 labels)	
		
