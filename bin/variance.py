import numpy
import math
import os

def range_var(name,dataset, lc=0):
	if not os.path.exists("results/variance/"+name):
		ac_var = 0
		range_var =[]
		if lc == 0:
			#Calculate label cardinality of dataSet
			ac = 0
			for instance in dataset:
				ac+ = numpy.count_nonzero(instance)
			lc=(ac_var / len(dataset))
	
		for instance in dataset:
			o = numpy.count_nonzero(instance)
			ac_var += pow(o-lc,2)
		var = math.ceil(math.sqrt(ac_var / len(dataset)))
		lc = round(lc)
		
		for i in range(lc-var,lc+var+1):
			with file("results/variance/"+name, "a") as mf:
				f.write(i)
				f.write(',')
	return




