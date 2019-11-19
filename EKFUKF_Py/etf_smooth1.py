import numpy as np
from numpy.linalg import inv, solve

from ekf_predict1 import ekf_predict1
from ekf_update1 import ekf_update1

def etf_smooth1(M, P, Y, A=None, Q=None, ia=None, W=None, aparam=None, H=None, R=None, h=None, V=None, hparam=None, same_p_a=1, same_p_h=1):
	"""
	ETF_SMOOTH1  Smoother based on two extended Kalman filters
	
	Syntax:
	  [M,P] = ETF_SMOOTH1(M,P,Y,A,Q,ia,W,aparam,H,R,h,V,hparam,same_p_a,same_p_h)
	
	In:
	  M - NxK matrix of K mean estimates from Kalman filter
	  P - NxNxK matrix of K state covariances from Kalman Filter
	  Y - Measurement vector
	  A - Derivative of a() with respect to state as
	      matrix, inline function, function handle or
	      name of function in form A(x,param)       (optional, default eye())
	  Q - Process noise of discrete model           (optional, default zero)
	 ia - Inverse prediction function as vector,
	      inline function, function handle or name
	      of function in form ia(x,param)           (optional, default inv(A(x))*X)
	  W - Derivative of a() with respect to noise q
	      as matrix, inline function, function handle
	      or name of function in form W(x,param)    (optional, default identity)
	  aparam - Parameters of a. Parameters should be a single cell array, vector or a matrix
	          containing the same parameters for each step or if different parameters
	          are used on each step they must be a cell array of the format
	          { param_1, param_2, ...}, where param_x contains the parameters for
	          step x as a cell array, a vector or a matrix.   (optional, default empty)
	  H  - Derivative of h() with respect to state as matrix,
	       inline function, function handle or name
	       of function in form H(x,param)
	  R  - Measurement noise covariance.
	  h  - Mean prediction (measurement model) as vector,
	       inline function, function handle or name
	       of function in form h(x,param).  (optional, default H(x)*X)
	  V  - Derivative of h() with respect to noise as matrix,
	       inline function, function handle or name
	       of function in form V(x,param).  (optional, default identity)
	  hparam - Parameters of h. See the description of aparam for the format of
	            parameters.                  (optional, default aparam)
	  same_p_a - If 1 uses the same parameters 
	             on every time step for a    (optional, default 1) 
	  same_p_h - If 1 uses the same parameters 
	             on every time step for h    (optional, default 1) 
	
	Out:
	  M - Smoothed state mean sequence
	  P - Smoothed state covariance sequence
	  
	Description:
	  Two filter nonlinear smoother algorithm. Calculate "smoothed"
	  sequence from given extended Kalman filter output sequence
	  by conditioning all steps to all measurements.
	
	Example:
	  [...]
	
	See also:
	  ERTS_SMOOTH1, EKF_PREDICT1, EKF_UPDATE1, EKF_PREDICT2, EKF_UPDATE2

	History:
	  02.08.2007 JH Changed the name to etf_smooth1
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

	M = M.copy()
	P = P.copy()

	m_1 = M.shape[0]

	# Run the backward filter
	BM = np.zeros(M.shape)
	BP = np.zeros(P.shape)
	#fm = zeros(size(M,1),1)
	#fP = 1e12*eye(size(M,1))
	fm = M[-1]
	fP = P[-1]
	BM[-1] = fm
	BP[-1] = fP
	
	for k in range(m_1-2,-1,-1):
		if hparam is None:
			hparams = None
		elif same_p_h:
			hparams = hparam
		else:
			hparams = hparam[k]
		
		if aparam is None:
			aparams = None
		elif same_p_a:
			aparams = aparam
		else:
			aparams = aparam[k]
		
		
		fm, fP, *_ = ekf_update1(fm, fP, Y[k+1], H, R, h, V, hparams)

		# Backward prediction
		if A is None:
			IA = None
		elif type(A) == np.ndarray:
			IA = inv(A)
		elif type(A)==str or callable(A):
			IA = inv(A(fm,aparams))
		else:
			IA = inv(A(fm,aparams))
		
		
		if W is None:
			if Q is not None:
				B = np.eye(M.shape[1])
			else:
				B = np.eye(M.shape[1])
		
		elif type(W) == np.ndarray:
			B = W
		elif type(W)==str or callable(W):
			B = W(fm,aparams)
		else:
			B = W(fm,aparams)
		    
		IQ = IA @ B @ Q @ B.T @ IA.T

		fm, fP = ekf_predict1(fm, fP, IA, IQ, ia, None, aparams)

		BM[k] = fm
		BP[k] = fP
	

	# Combine estimates
	for k in range(m_1-1):
		tmp = inv(inv(P[k]) + inv(BP[k]))
		M[k] = tmp @ (solve(P[k], M[k]) + solve(BP[k], BM[k]))
		P[k] = tmp
	
	return M, P