import numpy as np
from gh_transform import gh_transform

def ghkf_predict(M, P, f=None, Q=None, f_param=None, p=10):
	"""
	GHKF_PREDICT - Gauss-Hermite Kalman filter prediction step
	
	Syntax:
		[M,P] = GHKF_PREDICT(M,P,[f,Q,f_param,p])
	
	In:
		M - Nx1 mean state estimate of previous step
		P - NxN state covariance of previous step
		f - Dynamic model function as a matrix A defining
				linear function f(x) = A*x, inline function,
				function handle or name of function in
				form f(x,param)                   (optional, default eye())
		Q - Process noise of discrete model   (optional, default zero)
		f_param - Parameters of f               (optional, default empty)
		p - Degree of approximation (number of quadrature points)
	
	Out:
		M - Updated state mean
		P - Updated state covariance
	
	Description:
		Perform additive form Gauss-Hermite Kalman Filter prediction step.
	
		Function f(.) should be such that it can be given a
		DxN matrix of N sigma Dx1 points and it returns 
		the corresponding predictions for each sigma
		point. 
	
	See also:
		GHKF_UPDATE, GHRTS_SMOOTH, GH_TRANSFORM

	History:
		Aug 5,  2010 - Renamed from 'gh_predict' to 'ghkf_predict' (asolin)

	Copyright (C) 2009 Hartikainen, Särkkä, Solin

	$Id: gh_predict.m,v 1.2 2009/07/01 06:34:40 ssarkka Exp $

	This software is distributed under the GNU General Public
	Licence (version 2 or later); please refer to the file
	Licence.txt, included with the software, for details.
	"""

	m_ = M.shape[0]

	# Apply defaults
	if f is None:
		f = np.eye(m_)
	if Q is None:
		Q = np.zeros(m_)
	
	# Do transform and add process noise
	M, P, *_ = gh_transform(M, P, f, f_param, p)
	P += Q

	return M, P

