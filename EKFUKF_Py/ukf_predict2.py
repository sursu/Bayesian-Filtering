import numpy as np

from ut_transform import ut_transform

def ukf_predict2(M, P, f=None, Q=None, f_param=None, alpha=None, beta=None, kappa=None, mat=None):
	"""
	UKF_PREDICT2  Augmented (state and process noise) UKF prediction step
	
	Syntax:
	  [M,P] = UKF_PREDICT2(M,P,a,Q,[param,alpha,beta,kappa])
	
	In:
	  M - Nx1 mean state estimate of previous step
	  P - NxN state covariance of previous step
	  f - Dynamic model function as inline function,
	      function handle or name of function in
	      form a([x;w],param)
	  Q - Non-singular covariance of process noise w
	  f_param - Parameters of f               (optional, default empty)
	  alpha - Transformation parameter      (optional)
	  beta  - Transformation parameter      (optional)
	  kappa - Transformation parameter      (optional)
	  mat   - If 1 uses matrix form         (optional, default 0)
	
	Out:
	  M - Updated state mean
	  P - Updated state covariance
	
	Description:
	  Perform augmented form Unscented Kalman Filter prediction step
	  for model
	
	   x[k+1] = a(x[k],w[k],param)
	
	  Function a should be such that it can be given
	  DxN matrix of N sigma Dx1 points and it returns 
	  the corresponding predictions for each sigma
	  point. 
	
	See also:
	  UKF_PREDICT1, UKF_UPDATE1, UKF_UPDATE2, UKF_PREDICT3, UKF_UPDATE3,
	  UT_TRANSFORM, UT_WEIGHTS, UT_MWEIGHTS, UT_SIGMAS

	Copyright (C) 2003-2006 Simo S�rkk�
	
	$Id$
	
	This software is distributed under the GNU General Public
	Licence (version 2 or later); please refer to the file
	Licence.txt, included with the software, for details.
	"""


	# Do transform and add process noise
	n = Q.shape[0]
	p = P.shape[0]

	MA = np.vstack([M, np.zeros((n,1))])
	PA = np.zeros((n+p, n+p))
	PA[:p,:p] = P
	PA[p:,p:] = Q
	tr_param = [alpha, beta, kappa, mat]
	M, P, *_ = ut_transform(MA, PA, f, f_param, tr_param)

	return M, P
