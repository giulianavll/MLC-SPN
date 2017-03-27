import subprocess
import logging
import re
import shutil

INPUT_PATH = 'data/'
INPUTQ_PATH = 'data/queries'
OUTPUT_PATH = 'results/models_l/'

def learn_Libra(n_file,new_nfile, parameters):
	#Generate a spn using idspn in resulsts/models_l
	#params 
	likelihood = -99999
	args = ['libra','idspn','-i',INPUT_PATH+n_file+'.train','-o',OUTPUT_PATH+new_nfile+'.spn','-k',str(parameters['k']),'-ext',str(parameters['ext']),'-ps',str(parameters['ps']),'-f']
	p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	if out:
		likelihood = float(re.findall(r"[-+]?\d*\.\d+|\d+",str(out))[0])
		print(new_nfile)
		print(likelihood)
	if err:
		print("error of libra: "+ n_file + " ")
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
				if nfile=='yeast1' and ((i==1 and j==3 and k>0)or (i>1)):
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

	def inference_mg(eqname, spn):
		#libra spquery -m msweb-mt.spn -q msqweb.q -ev msweb.ev
		args=['libra','spquery','-m', OUTPUT_PATH+eqname+'.spn','-q',INPUTQ_PATH+query+'.q','-ev',INPUTQ_PATH+eqname+'.ev']
		p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = p.communicate()
		if out:
			sout = out.decode('ascii')
			lout = sout.split('\n')
			lp=[]
			for o in lout:
				if o.find(avg)==-1 and not o=='':
					lp.append(float(o))
			print(lp)
		if err:
			print("error of libra: "+ n_file + " ")
			print(err) 
		return lp


