import numpy as np
from numpy.linalg import solve

from gauss_pdf import gauss_pdf

def ekf_update1(M, P, y, H, R, h=None, V=None, param=None, nargout=2):
	"""
	EKF_UPDATE1  1st order Extended Kalman Filter update step
	
	Syntax:
	  [M,P,K,MU,S,LH] = EKF_UPDATE1(M,P,Y,H,R,[h,V,param])
	
	In:
	  M  - Nx1 mean state estimate after prediction step
	  P  - NxN state covariance after prediction step
	  Y  - Dx1 measurement vector.
	  H  - Derivative of h() with respect to state as matrix,
	       inline function, function handle or name
	       of function in form H(x,param)
	  R  - Measurement noise covariance.
	  h  - Mean prediction (innovation) as vector,
	       inline function, function handle or name
	       of function in form h(x,param).               (optional, default H(x)*X)
	  V  - Derivative of h() with respect to noise as matrix,
	       inline function, function handle or name
	       of function in form V(x,param).               (optional, default identity)
	  param - Parameters of h                            (optional, default empty)
	
	Out:
	  M  - Updated state mean
	  P  - Updated state covariance
	  K  - Computed Kalman gain
	  MU - Predictive mean of Y
	  S  - Predictive covariance of Y
	  LH - Predictive probability (likelihood) of measurement.
	  
	Description:
	  Extended Kalman Filter measurement update step.
	  EKF model is
	
	    y[k] = h(x[k],r),   r ~ N(0,R)
	
	See also:
	  EKF_PREDICT1, EKF_PREDICT2, EKF_UPDATE2, DER_CHECK,
	  LTI_DISC, KF_UPDATE, KF_PREDICT

	Copyright (C) 2002-2006 Simo S�rkk�
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	# Apply defaults
	if V is None:
		V = np.eye(R.shape[0])

	# Evaluate matrices
	if type(H) == np.ndarray:
		pass
	elif type(H) or callable(H):
		H = H(M,param)
	else:
		H = H(M,param)

	if h is None:
		MU = H@M
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
	S = (V@R@V.T + H@P@H.T)
	K = solve(S.T, H@P.T).T
	M += K @ (y-MU)
	P -= K@S@K.T

	if nargout > 5:
		LH, _ = gauss_pdf(y, MU, S)

		return M, P, K, MU, S, LH

	return M, P, K, MU, S
