from numpy import zeros, vstack
from numpy.linalg import solve

from ut_transform import ut_transform



def urts_smooth2(M, P, f, Q, f_param=None, alpha=None, beta=None, kappa=None, mat=0, same_p=True):
	"""
	URTS_SMOOTH2  Augmented form Unscented Rauch-Tung-Striebel smoother
	
	Syntax:
	  [M,P,S] = URTS_SMOOTH2(M,P,f,Q,[f_param,alpha,beta,kappa,mat,same_p])
	
	In:
	  M - NxK matrix of K mean estimates from Unscented Kalman filter
	  P - NxNxK matrix of K state covariances from Unscented Kalman Filter
	  f - Dynamic model function as inline function,
	      function handle or name of function in
	      form a([xw],param)
	  Q - Non-singular covariance of process noise w
	  f_param - Parameters of a. Parameters should be a single cell array,
	          vector or a matrix containing the same parameters for each
	          step, or if different parameters are used on each step they
	          must be a cell array of the format { param_11, param_12, ...},
	          where param_1x contains the parameters for step x as a cell array,
	          a vector or a matrix.   (optional, default empty)
	  alpha - Transformation parameter      (optional)
	  beta  - Transformation parameter      (optional)
	  kappa - Transformation parameter      (optional)
	  mat   - If 1 uses matrix form         (optional, default 0)
	  same_p - If 1 uses the same parameters 
	           on every time step      (optional, default 1)   
	
	Out:
	  K - Smoothed state mean sequence
	  P - Smoothed state covariance sequence
	  D - Smoother gain sequence
	  
	Description:
	  Unscented Rauch-Tung-Striebel smoother algorithm. Calculate
	  "smoothed" sequence from given Kalman filter output sequence by
	  conditioning all steps to all measurements.
	
	Example:
	  [...]
	
	See also:
	  URTS_SMOOTH1, UKF_PREDICT2, UKF_UPDATE2

	Copyright (C) 2006 Simo S�rkk�
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	M = M.copy()
	P = P.copy()

	m_1, m_2 = M.shape[:2]
	q_ = Q.shape[0]
	p_ = P.shape[1]

	# Run the smoother
	D = zeros((m_1,m_2,m_2))
	for k in range(m_1-2,-1,-1):
		if f_param is None:
			params = None
		elif same_p:
			params = f_param
		else:
			params = f_param[k]
		
		
		MA = vstack([M[k], zeros((q_,1))])
		PA = zeros((p_+q_,p_+q_))
		PA[:p_,:p_] = P[k]
		PA[p_:,p_:] = Q
		
		tr_param = [alpha, beta, kappa, mat]
		m_pred, P_pred, C, *_ = ut_transform(MA, PA, f, params, tr_param)
		C = C[:m_2]
		
		D[k] = solve(P_pred.T, C.T).T
		M[k] += D[k] @ (M[k+1] - m_pred)
		P[k] += D[k] @ (P[k+1] - P_pred) @ D[k].T
	

	return M, P, D