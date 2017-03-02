import subprocess

def learn_Libra(n_file):
	args=['libra','idspn','-i','DataSet/'+n_file+'.train','-o','Models/'+n_file+'.spn','-k','10','-ext','15','-l','0.2','-vth','0.001','-ps','20','-l1','20','-f']
	p = subprocess.Popen(args) 
	return

