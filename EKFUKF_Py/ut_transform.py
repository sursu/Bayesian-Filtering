import numpy as np

from ut_mweights import ut_mweights
from ut_weights import ut_weights
from ut_sigmas import ut_sigmas

def ut_transform(M, P, g, g_param=None, tr_param=None):
	"""
	UT_TRANSFORM  Perform unscented transform
	
	Syntax:
	  [mu,S,C,X,Y,w] = UT_TRANSFORM(M,P,g,g_param,tr_param)
	
	In:
	  M - Random variable mean (Nx1 column vector)
	  P - Random variable covariance (NxN pos.def. matrix)
	  g - Transformation function of the form g(x,param) as
	      matrix, inline function, function name or function reference
	  g_param - Parameters of g               (optional, default empty)
	  tr_param - Parameters of the transformation as:       
	      alpha = tr_param{1} - Transformation parameter      (optional)
	      beta  = tr_param{2} - Transformation parameter      (optional)
	      kappa = tr_param{3} - Transformation parameter      (optional)
	      mat   = tr_param{4} - If 1 uses matrix form         (optional, default 0)
	      X     = tr_param{5} - Sigma points of x
	      w     = tr_param{6} - Weights as cell array {mean-weights,cov-weights,c}
	
	Out:
	  mu - Estimated mean of y
	   S - Estimated covariance of y
	   C - Estimated cross-covariance of x and y
	   X - Sigma points of x
	   Y - Sigma points of y
	   w - Weights as cell array {mean-weights,cov-weights,c}
	
	Description:
	  ...
	  For default values of parameters, see UT_WEIGHTS.
	
	See also
	  UT_WEIGHTS UT_MWEIGHTS UT_SIGMAS

	Copyright (C) 2006 Simo S�rkk�
	              2010 Jouni Hartikainen
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""
	m_ = M.shape[0]

	# Apply defaults
	alpha, beta, kappa, mat, X, w = tr_param + [None,] * (6 - len(tr_param))
	  
	# Calculate sigma points
	if w is not None:
		WM = w[0]
		c  = w[2]
		if mat:
			W  = w[1]
		else:
			WC = w[1]
	elif mat:
		WM, W, c = ut_mweights(m_, alpha, beta, kappa)
		X = ut_sigmas(M,P,c)
		w = [WM, W, c]
	else:
		WM, WC, c = ut_weights(m_, alpha, beta, kappa)
		X = ut_sigmas(M, P, c)
		w = [WM, WC, c]
	
	
	# Propagate through the function
	if type(g)==np.ndarray:
		Y = g@X
	elif type(g)==str or callable(g):
		Y = g(X, g_param)
	else:
		Y = g(X, g_param)
		
	if mat:
		mu = Y@WM
		S  = Y@W@Y.T
		C  = X@W@Y.T
	else:
		mu = Y@WM[:,None]
		S  = WC * (Y-mu) @ (Y-mu).T
		C  = WC * (X[:M.shape[0]]-M) @ (Y-mu).T

	return mu, S, C, X, Y, w
