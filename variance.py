import numpy
import math
import os
import sys
sys.path.append('../')
from bin import dataset as ds

def range_var(name,nlabels, lc=0):
	if not os.path.exists('results/variance/'+name):
		dataset = ds.csv_2_numpy(name+'.train')
		ac_var = 0
		range_var =[]
		if lc == 0:
			#Calculate label cardinality of dataSet
			ac = 0
			for instance in dataset:
				ac = ac + numpy.count_nonzero(instance[:nlabels])
			lc=(ac_var / len(dataset))

		for instance in dataset:
			o = numpy.count_nonzero(instance[:nlabels])
			ac_var = ac_var + pow(o-lc,2)
		var = math.ceil(math.sqrt(ac_var / len(dataset)))
		lc = round(lc)
		for i in range(lc-var,lc+var+1):
			with open("results/variance/"+name, "a") as mf:
				mf.write(str(i))
				if not i==lc+var:
					mf.write(',')
	return

range_var('health0',32,1.644)
range_var('health1',32,1.644)
range_var('health2',32,1.644)
range_var('health3',32,1.644)
range_var('health4',32,1.644)

range_var('human0',14,1.185)
range_var('human1',14,1.185)
range_var('human2',14,1.185)
range_var('human3',14,1.185)
range_var('human4',14,1.185)

range_var('plant0',12,1.078)
range_var('plant1',12,1.078)
range_var('plant2',12,1.078)
range_var('plant3',12,1.078)
range_var('plant4',12,1.078)

range_var('scene0',6,1.073)
range_var('scene1',6,1.073)
range_var('scene2',6,1.073)
range_var('scene3',6,1.073)
range_var('scene4',6,1.073)

range_var('yeast0',14,4.237)
range_var('yeast1',14,4.237)
range_var('yeast2',14,4.237)
range_var('yeast3',14,4.237)
range_var('yeast4',14,4.237)