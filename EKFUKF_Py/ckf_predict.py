import numpy as np

from ckf_transform import ckf_transform

def ckf_predict(M, P, f=None, Q=None, f_param=None):
	"""
	CKF_PREDICT - Cubature Kalman filter prediction step
	
	Syntax:
		[M,P] = CKF_PREDICT(M,P,[f,Q,f_param])
	
	In:
		M - Nx1 mean state estimate of previous step
		P - NxN state covariance of previous step
		f - Dynamic model function as a matrix A defining
				linear function f(x) = A*x, inline function,
				function handle or name of function in
				form f(x,param)                   (optional, default eye())
		Q - Process noise of discrete model   (optional, default zero)
		f_param - Parameters of f               (optional, default empty)
	
	Out:
		M - Updated state mean
		P - Updated state covariance
	
	Description:
		Perform additive form spherical-radial cubature Kalman filter (CKF)
		prediction step.
	
		Function f(.) should be such that it can be given a
		DxN matrix of N sigma Dx1 points and it returns 
		the corresponding predictions for each sigma
		point. 
	
	See also:
		CKF_UPDATE, CRTS_SMOOTH, CKF_TRANSFORM, SPHERICALRADIAL
	
	References:
		Arasaratnam and Haykin (2009). Cubature Kalman Filters.
			IEEE Transactions on Automatic Control, vol. 54, no. 5, pp.1254-1269

	Copyright (c) 2010 Arno Solin
	
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
	M, P, *_ = ckf_transform(M, P, f, f_param)
	P += Q

	return M, P