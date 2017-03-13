import subprocess
import logging
import re
import shutil

INPUT_PATH = 'data/'
OUTPUT_PATH = 'results/models_l/'

def learn_Libra(n_file,new_nfile, parameters):
	#Generate a spn using idspn in resulsts/models_l
	#params 
	likelihood = -99999
	args = ['libra','idspn','-i',INPUT_PATH+n_file+'.train','-o',OUTPUT_PATH+new_nfile+'.spn','-k',str(parameters['k']),'-ext',str(parameters['ext']),'-ps',str(parameters['ps']),'-f']
	p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	if out:
		likelihood = float(re.findall("\d+\.\d+",str(out))[0])
	if err:
		print("standard error of of libra: "+ n_file + " ")
		print(err) 
	return likelihood

def create_IDnetworks(nfile, v_parameters=[2,5,10,20]):
	#Comparations of parameters
	likelihood = -99999
	best_spn = ''
	logging.info('-- Train SPN with diferent parameters --')
	logging.info('This process may take a few minutes.......')
	for i in range(0,len(v_parameters)):
		for j in range(0,len(v_parameters)):
			for k in range(0,len(v_parameters)):
				if i==0 and j==0 and k==0:
					i=1
					j=2
					k=2
					print(i)

				parameters={}
				parameters['ext']= v_parameters[i]
				parameters['k']= v_parameters[j]
				parameters['ps']= v_parameters[k]
				new_name= nfile+str(i)+str(j)+str(k)
				print(new_name)
				t_likelihood = learn_Libra(nfile,new_name,parameters)
				if t_likelihood > likelihood :
					if best_spn != '':
						shutil.rmtree(OUTPUT_PATH+ best_spn+'.spn')
					likelihood = t_likelihood
					best_spn = new_name
				else:
					shutil.rmtree(OUTPUT_PATH+ new_name+'.spn')


	logging.info('---Found the best network---')				
	return best_spn


