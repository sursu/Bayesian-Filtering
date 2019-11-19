from numpy import vstack, zeros
import numpy as np

from ut_transform import ut_transform

def ukf_predict3(M, P, f=None, Q=None, R=None, f_param=None, alpha=None, beta=None, kappa=None, mat=None):
	"""
	UKF_PREDICT3  Augmented (state, process and measurement noise) UKF prediction step
	
	Syntax:
	  [M,P,X,w] = UKF_PREDICT3(M,P,f,Q,R,f_param,alpha,beta,kappa)
	
	In:
	  M - Nx1 mean state estimate of previous step
	  P - NxN state covariance of previous step
	  f - Dynamic model function as inline function,
	      function handle or name of function in
	      form a([x;w],param)
	  Q - Non-singular covariance of process noise w
	  R - Measurement covariance.
	  f_param - Parameters of f               (optional, default empty)
	  alpha - Transformation parameter      (optional)
	  beta  - Transformation parameter      (optional)
	  kappa - Transformation parameter      (optional)
	  mat   - If 1 uses matrix form         (optional, default 0)
	
	Out:
	  M - Updated state mean
	  P - Updated state covariance
	  X - Sigma points of x
	  w - Weights as cell array {mean-weights,cov-weights,c}
	
	Description:
	  Perform augmented form Unscented Kalman Filter prediction step
	  for model
	
	   x[k+1] = a(x[k],w[k],param)
	
	  Function a should be such that it can be given
	  DxN matrix of N sigma Dx1 points and it returns 
	  the corresponding predictions for each sigma
	  point. 
	
	See also:
	  UKF_PREDICT1, UKF_UPDATE1, UKF_PREDICT2, UKF_UPDATE2, UKF_UPDATE3
	  UT_TRANSFORM, UT_WEIGHTS, UT_MWEIGHTS, UT_SIGMAS 

	Copyright (C) 2003-2006 Simo S�rkk�
	Copyright (C) 2007 Jouni Hartikainen
	
	$Id$
	
	This software is distributed under the GNU General Public
	Licence (version 2 or later); please refer to the file
	Licence.txt, included with the software, for details.
	"""

	# Apply defaults
	if mat is None:
		mat = 0

	i1 = P.shape[0]
	q_ = Q.shape[0]
	r_ = R.shape[0]
	i2 = i1 + q_

	# Do transform and add process and measurement noises
	MA = vstack([M, zeros((q_,1)), zeros((r_,1))])
	PA = zeros((i1+q_+r_,i1+q_+r_))

	PA[:i1,:i1] = P
	PA[i1:i2,i1:i2] = Q
	PA[i2:,i2:] = R
	
	tr_param = [alpha, beta, kappa, mat]
	M, P, C, X_s, X_pred, w = ut_transform(MA, PA, f, f_param, tr_param)
		
	# Save sigma points
	X = X_s
	X[:X_pred.shape[0]] = X_pred
	
	return M, P, X, w, C
	