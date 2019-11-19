import numpy as np

from ut_weights import ut_weights

def ut_mweights(n, alpha=None, beta=None, kappa=None):
	"""
	UT_MWEIGHTS - Generate matrix form unscented transformation weights
	
	Syntax:
	  [WM,W,c] = ut_mweights(n,alpha,beta,kappa)
	
	In:
	  n     - Dimensionality of random variable
	  alpha - Transformation parameter  (optional, default 0.5)
	  beta  - Transformation parameter  (optional, default 2)
	  kappa - Transformation parameter  (optional, default 3-size(X,1))
	
	Out:
	  WM - Weight vector for mean calculation
	   W - Weight matrix for covariance calculation
	   c - Scaling constant
	
	Description:
	  Computes matrix form unscented transformation weights.
	

	Copyright (C) 2006 Simo S�rkk�
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""
	
	WM, WC, c = ut_weights(n, alpha, beta, kappa)

	W = np.eye(len(WC)) - np.tile(WM,(1,len(WM)))
	W = W @ np.diag(WC) @ W.T

	return WM, W, c