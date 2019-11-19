import numpy as np
from numpy.linalg import solve

from ckf_transform import ckf_transform

def crts_smooth(M, P, f, Q, f_param=None, same_p=True):
	"""
	CRTS_SMOOTH - Additive form cubature Rauch-Tung-Striebel smoother

	Syntax:
		[M,P,D] = CKF_SMOOTH(M,P,a,Q,[param,same_p])

	In:
		M - NxK matrix of K mean estimates from Cubature Kalman filter
		P - NxNxK matrix of K state covariances from Cubature Kalman Filter
		f - Dynamic model function as a matrix F defining
			linear function f(x) = F*x, inline function,
			function handle or name of function in
			form f(x,param)                   (optional, default eye())
		Q - NxN process noise covariance matrix or NxNxK matrix
			of K state process noise covariance matrices for each step.
		f_param - Parameters of f. Parameters should be a single cell array,
				vector or a matrix containing the same parameters for each
				step, or if different parameters are used on each step they
				must be a cell array of the format { param_1, param_2, ...},
				where param_x contains the parameters for step x as a cell array,
				a vector or a matrix.   (optional, default empty)
		same_p - If 1 uses the same parameters 
				on every time step      (optional, default 1) 

	Out:
		M - Smoothed state mean sequence
		P - Smoothed state covariance sequence
		D - Smoother gain sequence
		
	Description:
		Cubature Rauch-Tung-Striebel smoother algorithm. Calculate
		"smoothed" sequence from given Kalman filter output sequence by
		conditioning all steps to all measurements. Uses the spherical-
		radial cubature rule.

	Example:
		m = m0
		P = P0
		MM = zeros(size(m,1),size(Y,2))
		PP = zeros(size(m,1),size(m,1),size(Y,2))
		for k=1:size(Y,2)
		[m,P] = ckf_predict(m,P,f,Q)
		[m,P] = ckf_update(m,P,Y(:,k),h,R)
		MM(:,k) = m
		PP(:,:,k) = P
		end
		[SM,SP] = crts_smooth(MM,PP,f,Q)

	See also:
		CKF_PREDICT, CKF_UPDATE, SPHERICALRADIAL

	Copyright (c) 2010 Arno Solin

	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""
	M = M.copy()
	P = P.copy()

	# Apply defaults
	m_1, m_2 = M.shape[:2]

	if f is None:
		f = np.eye(m_2)

	if Q is None:
		Q = np.zeros(m_2)


	# Extend Q if NxN matrix
	if len(Q.shape)==2:
		Q = np.tile(Q, (m_1,1,1))


	# Run the smoother
	D = np.zeros((m_1,m_2,m_2))
	if f_param is None:
		for k in range(m_1-2,-1,-1):
			m_pred, P_pred, C, *_ = ckf_transform(M[k], P[k], f)
			P_pred += Q[k]
			D[k] = solve(P_pred.T, C.T).T
			M[k] += D[k] @ (M[k+1] - m_pred)
			P[k] +=  D[k] @ (P[k+1] - P_pred) @ D[k].T  
	else:
		for k in range(m_1-2,-1,-1):
			if f_param is None:
				params = None
			elif same_p:
				params = f_param
			else:
				params = f_param[k]
		
			m_pred, P_pred, C, *_ = ckf_transform(M[k], P[k], f, params)
			P_pred += Q[k]
			D[k] = solve(P_pred.T, C.T).T
			M[k] += D[k] @ (M[k+1] - m_pred)
			P[k] += D[k] @ (P[k+1] - P_pred) @ D[k].T

	return M, P, D