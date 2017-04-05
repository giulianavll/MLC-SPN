# MLC-SPN
Multi-Label Classification based on Sum-Product Networks

## overview

Use the implementations:

[LearnSPN](http://homes.cs.washington.edu/~pedrod/papers/mlc13.pdf) and SPN-AL, SPN-AC as presented in:  

	_A. Vergari, N. Di Mauro, andF. Esposito_   
	**Simplifying, Regularizing and Strengthening Sum-Product Network Structure Learning** at ECML-PKDD 2015.
	
[Libra toolkit](http://libra.cs.uoregon.edu/) presented in:

	_D. Lowd and A. Rooshenas_
	**The Libra Toolkit for Probabilistic Models** in Journal of Machine Learning Research 2015.


## requirements
MLC-SPN is build upon python3 [numpy](http://www.numpy.org/),
[sklearn](http://scikit-learn.org/stable/),
[scipy](http://www.scipy.org/), [numba](http://numba.pydata.org/), [matplotlib](http://matplotlib.org/) and [theano](http://deeplearning.net/software/theano/), and the [LibraTollkit](http://libra.cs.uoregon.edu/doc/manual.pdf)

## usage

To run the algorithm execute :
python3 classify.py -dataset flags -nl 7 -ml ac  -ap pcc --folds

python3 classify.py -dataset flags0 -nl 7 -ml ac  -ap pcc 


[-dataset] Dataset name (Only use datasets whose contents are binary values)
[-nl] Labels number in the dataset
[-ml] SPN learning methods (al,ac,id)
[-ap] Multi-Label classification approach (br,cc, pcc,mpe,lp)
[--folds] Find and process the 5-folds for training and classification (dataset+nfold)
[--bagg] Use bagging, only for the methods al and ac

Several datasets are provided in the `data/` folder.