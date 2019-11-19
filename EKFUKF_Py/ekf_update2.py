import numpy as np
from numpy.linalg import solve

from gauss_pdf import gauss_pdf

def ekf_update2(M, P, y, H, H_xx, R, h=None, V=None, param=None, nargout=6):
	"""
	EKF_UPDATE2  2nd order Extended Kalman Filter update step
	
	Syntax:
	  [M,P,K,MU,S,LH] = EKF_UPDATE2(M,P,Y,H,H_xx,R,[h,V,param])
	
	In:
	  M  - Nx1 mean state estimate after prediction step
	  P  - NxN state covariance after prediction step
	  Y  - Dx1 measurement vector.
	  H  - Derivative of h() with respect to state as matrix,
	       inline function, function handle or name
	       of function in form H(x,param)
	  H_xx - DxNxN Hessian of h() with respect to state as matrix,
	         inline function, function handle or name of function
	         in form H_xx(x,param) 
	  R  - Measurement noise covariance.
	  h  - Mean prediction (measurement model) as vector,
	       inline function, function handle or name
	       of function in form h(x,param).                 (optional, default H(x)*X)
	  V  - Derivative of h() with respect to noise as matrix,
	       inline function, function handle or name
	       of function in form V(x,param).                 (optional, default identity)
	  param - Parameters of h                              (optional, default empty)
	
	Out:
	  M  - Updated state mean
	  P  - Updated state covariance
	  K  - Computed Kalman gain
	  MU - Predictive mean of Y
	  S  - Predictive covariance Y
	  LH - Predictive probability (likelihood) of measurement.
	  
	Description:
	  Extended Kalman Filter measurement update step.
	  EKF model is
	
	    y[k] = h(x[k],r),   r ~ N(0,R)
	
	See also:
	  EKF_PREDICT1, EKF_UPDATE1, EKF_PREDICT2, DER_CHECK, LTI_DISC, 
	  KF_UPDATE, KF_PREDICT

	Copyright (C) 2002-2006 Simo S�rkk�
	Copyright (C) 2007 Jouni Hartikainen
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""


	# Apply defaults

	if V is None:
		V = np.eye(R.shape[0])

	#
	# Evaluate matrices
	#
	if type(H) == np.ndarray:
		pass
	elif type(H)==str or callable(H):
		H = H(M,param)
	else:
		H = H(M,param)
	
	if type(H_xx) == np.ndarray:
		pass
	elif type(H_xx)==str or callable(H_xx):
		H_xx = H_xx(M,param)
	else:
		H_xx = H_xx(M,param)

	if h is None:
		MU = H*M
	elif type(h) == np.ndarray:
		MU = h
	elif type(h)==str or callable(h):
		MU = h(M,param)
	else:
		MU = h(M,param)

	if type(V) == np.ndarray:
		pass
	elif type(V)==str or callable(V):
		V = V(M,param)
	else:
		V = V(M,param)

	
	# Update step
	v = y - MU
	for i in range(H_xx.shape[0]):
		v[i] -= 0.5*np.trace(H_xx[i]@P)
	
	S = V@R@V.T + H@P@H.T
	for i in range(H_xx.shape[0]):
		for j in range(H_xx.shape[0]):
			H_i = H_xx[i]
			H_j = H_xx[j]
			S[i,j] += 0.5*np.trace(H_i@P@H_j@P)
	K = solve(S.T, H@P.T).T
	M += K @ v
	P -= K@S@K.T

	if nargout > 5:
		LH = gauss_pdf(y, MU, S)

	return M, P, K, MU, S, LH
