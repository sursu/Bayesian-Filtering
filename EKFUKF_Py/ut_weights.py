import numpy as np

def ut_weights(n, alpha=1, beta=0, kappa=None):
	"""
	UT_WEIGHTS - Generate unscented transformation weights
	
	Syntax:
	  [WM,WC,c] = ut_weights(n,alpha,beta,kappa)
	
	In:
	  n     - Dimensionality of random variable
	  alpha - Transformation parameter  (optional, default 0.5)
	  beta  - Transformation parameter  (optional, default 2)
	  kappa - Transformation parameter  (optional, default 3-n)
	
	Out:
	  WM - Weights for mean calculation
	  WC - Weights for covariance calculation
	   c - Scaling constant
	
	Description:
	  Computes unscented transformation weights.
	
	See also UT_MWEIGHTS UT_TRANSFORM UT_SIGMAS
	

	Copyright (C) 2006 Simo S�rkk�
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	# Apply default values
	if alpha is None:
		alpha = 1
	if beta is None:
		beta = 0
	if kappa is None:
		kappa = 3 - n
		

	# Compute the normal weights 
	lmda = alpha**2 * (n + kappa) - n
		
	WM, WC = np.zeros((2, 2*n+1))
	WM[0] = lmda / (n + lmda)
	WC[0] = lmda / (n + lmda) + (1 - alpha**2 + beta)
	for j in range(1, 2*n+1):
		wm = 1 / (2 * (n + lmda))
		WM[j] = wm
		WC[j] = wm

	c = n + lmda

	return WM, WC, c