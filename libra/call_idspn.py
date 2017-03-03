import subprocess
INPUT_PATH = 'data/'
OUTPUT_PATH = 'results/models_l/'

def learn_Libra(n_file):
	#Generate a spn using idspn in resulsts/models_l
	#params 
	args=['libra','idspn','-i',INPUT_PATH+n_file+'.train','-o',OUTPUT_PATH+n_file+'.spn','-k','10','-ext','5','-l','0.2','-vth','0.001','-ps','20','-l1','20']
	p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	if out:
    	print("standard output of libra: "+ n_file + " ")
    	print(out)
	if err:
    	print("standard error of of libra: "+ n_file + " ")
    	print(err) 
	return

