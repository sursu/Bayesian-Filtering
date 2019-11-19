import numpy as np    
from numpy.linalg import cholesky


def sphericalradial(f, m, P, param=None):
	"""
	SPHERICALRADIAL - ND scaled spherical-radial cubature rule
	
	Syntax:
		[I,x,W,F] = sphericalradial(f,m,P[,param])
	
	In:
		f - Function f(x,param) as inline, name or reference
		m - Mean of the d-dimensional Gaussian distribution
		P - Covariance of the Gaussian distribution
		param - Parameters for the function (optional)
	
	Out:
		I - The integral
		x - Evaluation points
		W - Weights
		F - Function values
	
	Description:
		Apply the spherical-radial cubature rule to integrals of form:
			int f(x) N(x | m,P) dx

	History:
		Aug 5, 2010 - Renamed from 'cubature' to 'sphericalradial' (asolin)

	Copyright (c) 2010 Arno Solin
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	## Spherical-radial cubature rule

	# The dimension of m is
	n = m.shape[0]
	
	# Evaluation points (nx2n)
	x = np.hstack([np.eye(n), -np.eye(n)])
	
	# Scaling
	x = np.sqrt(n)*x
	x = cholesky(P)@x + np.tile(m, 2*n)

	# Evaluate the function at the points
	if type(f)==str or callable(f):
			if param is None:
					F = f(x)
			else:
					F = f(x,param)
	elif type(f)==np.ndarray:
			F = f@x
	else:
			if param is None:
					F = f(x)
			else:
					F = f(x,param)
	
	# The weights are
	W = 1/(2*n)
	
	# Return integral value
	I = W*np.sum(F,axis=1, keepdims=True)
	
	# Return weights
	W = W*np.ones(2*n)
	
	return I, x, W, F