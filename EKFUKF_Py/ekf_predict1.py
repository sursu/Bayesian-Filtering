import numpy as np


def ekf_predict1(M, P, A=None, Q=None, a=None, W=None, param=None):
	"""
	EKF_PREDICT1  1st order Extended Kalman Filter prediction step
	
	Syntax:
	  [M,P] = EKF_PREDICT1(M,P,[A,Q,a,W,param])
	
	In:
	  M - Nx1 mean state estimate of previous step
	  P - NxN state covariance of previous step
	  A - Derivative of a() with respect to state as
	      matrix, inline function, function handle or
	      name of function in form A(x,param)       (optional, default eye())
	  Q - Process noise of discrete model               (optional, default zero)
	  a - Mean prediction E[a(x[k-1],q=0)] as vector,
	      inline function, function handle or name
	      of function in form a(x,param)                (optional, default A(x)*X)
	  W - Derivative of a() with respect to noise q
	      as matrix, inline function, function handle
	      or name of function in form W(x,param)        (optional, default identity)
	  param - Parameters of a                           (optional, default empty)
	
	Out:
	  M - Updated state mean
	  P - Updated state covariance
	  
	Description:
	  Perform Extended Kalman Filter prediction step.
	
	See also:
	  EKF_UPDATE1, EKF_PREDICT2, EKF_UPDATE2, DER_CHECK,
	  LTI_DISC, KF_PREDICT, KF_UPDATE

	Copyright (C) 2002-2006 Simo S�rkk�
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	m_ = M.shape[0]

	# Apply defaults
	if A is None:
		A = np.eye(m_)
	if Q is None:
		Q = np.zeros(m_)
	if W is None:
		W = np.eye(m_,Q.shape[1])

	if type(A) == np.ndarray:
		pass
	elif type(A)==str or callable(A):
		A = A(M,param)
	else:
		A = A(M,param)


	# Perform prediction
	if a is None:
		M = A@M
	elif type(a) == np.ndarray:
		M = a
	elif type(a)==str or callable(a):
		M = a(M, param)
	else:
		M = a(M, param)


	if type(W) == np.ndarray:
		pass
	elif type(W)==str or callable(W):
		W = W(M, param)
	else:
		W = W(M,param)

	P = A @ P @ A.T + W @ Q @ W.T

	return M, P