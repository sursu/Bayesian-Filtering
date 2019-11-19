import numpy as np
from numpy.linalg import solve

def erts_smooth1(M, P, A=None, Q=None, a=None, W=None, param=None, same_p=True):
	"""
	ERTS_SMOOTH1  Extended Rauch-Tung-Striebel smoother
	
	Syntax:
	  [M,P,D] = ERTS_SMOOTH1(M,P,A,Q,[a,W,param,same_p])
	
	In:
	  M - NxK matrix of K mean estimates from Unscented Kalman filter
	  P - NxNxK matrix of K state covariances from Unscented Kalman Filter
	  A - Derivative of a() with respect to state as
	      matrix, inline function, function handle or
	      name of function in form A(x,param)                 (optional, default eye())
	  Q - Process noise of discrete model                       (optional, default zero)
	  a - Mean prediction E[a(x[k-1],q=0)] as vector,
	      inline function, function handle or name
	      of function in form a(x,param)                        (optional, default A(x)*X)
	  W - Derivative of a() with respect to noise q
	      as matrix, inline function, function handle
	      or name of function in form W(x,param)                (optional, default identity)
	  param - Parameters of a. Parameters should be a single cell array, vector or a matrix
	          containing the same parameters for each step or if different parameters
	          are used on each step they must be a cell array of the format
	          { param_1, param_2, ...}, where param_x contains the parameters for
	          step x as a cell array, a vector or a matrix.     (optional, default empty)
	  same_p - 1 if the same parameters should be
	           used on every time step                          (optional, default 1)
	                                  
	                        
	
	Out:
	  K - Smoothed state mean sequence
	  P - Smoothed state covariance sequence
	  D - Smoother gain sequence
	  
	Description:
	  Extended Rauch-Tung-Striebel smoother algorithm. Calculate
	  "smoothed" sequence from given Kalman filter output sequence by
	  conditioning all steps to all measurements.
	
	Example:
	
	See also:
	  EKF_PREDICT1, EKF_UPDATE1

	History:
	  04.05.2007 JH Added the possibility to pass different parameters for a and h
	                for each step.
	  2006       SS Initial version.  
	
	Copyright (C) 2006 Simo S�rkk�
	Copyright (C) 2007 Jouni Hartikainen
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	m_1, m_2 = M.shape[:2]

	# Apply defaults

	if A is None:
		A = np.eye(m_2)

	if Q is None:
		Q = np.zeros(m_2)

	if W is None:
		W = np.eye(m_2)


	# Extend Q if NxN matrix
	if len(Q.shape) < 3:
		Q = np.tile(Q,(m_1,1,1))


	# Run the smoother
	M = M.copy()
	P = P.copy()
	D = np.zeros((m_1,m_2,m_2))
	for k in range(m_1-2,-1,-1):
		if param is None:
			params = None  
		elif same_p:
			params = param
		else:
			params = param[k]

		# Perform prediction
		if a is None:
			m_pred = A@M[k]
		elif type(a)==np.ndarray:
			m_pred = a
		elif type(a)==str or callable(a):
			m_pred = a(M[k],params)
		else:
			m_pred = a(M[k],params)
		
		if type(A) == np.ndarray:
			F = A
		elif type(A) or callable(A):
			F = A(M[k],params)
		else:
			F = A(M[k],params)
		
		if type(W) == np.ndarray:
			B = W
		elif type(W) or callable(W):
			B = W(M[k],params)
		else:
			B = W(M[k],params)

		P_pred = F @ P[k] @ F.T + B @ Q[k] @ B.T
		C = P[k] @ F.T
		
		D[k] = solve(P_pred.T, C.T).T
		M[k] += D[k] @ (M[k+1] - m_pred)
		P[k] += D[k] @ (P[k+1] - P_pred) @ D[k].T

	return M, P, D