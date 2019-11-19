import numpy as np
from numpy.linalg import solve, inv

from ukf_update1 import ukf_update1
from ukf_predict2 import ukf_predict2

def utf_smooth1(M, P, Y, ia=None, Q=None, aparam=None, h=None, R=None, hparam=None, 
				alpha=None, beta=None, kappa=None, mat=0, same_p_a=1, same_p_h=1):
	"""
	UTF_SMOOTH1  Smoother based on two unscented Kalman filters
	
	Syntax:
	  [M,P] = UTF_SMOOTH1(M,P,Y,[ia,Q,aparam,h,R,hparam,,alpha,beta,kappa,mat,same_p_a,same_p_h])
	
	In:
	  M - NxK matrix of K mean estimates from Kalman filter
	  P - NxNxK matrix of K state covariances from Kalman Filter
	  Y - Measurement vector
	 ia - Inverse prediction as a matrix IA defining
	      linear function ia(xw) = IA*xw, inline function,
	      function handle or name of function in
	      form ia(xw,param)                         (optional, default eye())
	  Q - Process noise of discrete model           (optional, default zero)
	  aparam - Parameters of a                      (optional, default empty)
	  h  - Measurement model function as a matrix H defining
	       linear function h(x) = H*x, inline function,
	       function handle or name of function in
	       form h(x,param)
	  R  - Measurement noise covariance.
	  hparam - Parameters of h              (optional, default aparam)
	  alpha - Transformation parameter      (optional)
	  beta  - Transformation parameter      (optional)
	  kappa - Transformation parameter      (optional)
	  mat   - If 1 uses matrix form         (optional, default 0)
	  same_p_a - If 1 uses the same parameters 
	             on every time step for a   (optional, default 1)
	  same_p_h - If 1 uses the same parameters 
	             on every time step for h   (optional, default 1) 
	
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
	  UKF_PREDICT1, UKF_UPDATE1

	History:
	  02.08.2007 JH Changed the name to utf_smooth1
	  04.05.2007 JH Added the possibility to pass different parameters for a and h
	                for each step.
	  2006       SS Initial version.           

	Copyright (C) 2006 Simo S�rkk�
	              2007 Jouni Hartikainen
	
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
		
		fm, fP, *_ = ukf_update1(fm, fP, Y[k+1], h, R, hparams, alpha, beta, kappa, mat)
	
		# Backward prediction
		fm, fP = ukf_predict2(fm, fP, ia, Q, aparams)
		BM[k] = fm
		BP[k] = fP

	# Combine estimates
	for k in range(m_1-1):
		tmp = inv(inv(P[k]) + inv(BP[k]))
		M[k] = tmp @ (solve(P[k], M[k]) + solve(BP[k], BM[k]))
		P[k] = tmp

	return M, P