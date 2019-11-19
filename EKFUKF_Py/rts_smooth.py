# RTS_SMOOTH  

# Copyright (C) 2003-2006 Simo Särkkä
#
# $Id$
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

import numpy as np
from numpy.linalg import solve

def rts_smooth(M, P, A, Q):
	"""
	Rauch-Tung-Striebel smoother

	Syntax:
	[M,P,S] = RTS_SMOOTH(M,P,A,Q)

	In:
	M - NxK matrix of K mean estimates from Kalman filter
	P - NxNxK matrix of K state covariances from Kalman Filter
	A - NxN state transition matrix or NxNxK matrix of K state
		transition matrices for each step.
	Q - NxN process noise covariance matrix or NxNxK matrix
		of K state process noise covariance matrices for each step.

	Out:
	M - Smoothed state mean sequence
	P - Smoothed state covariance sequence
	D - Smoother gain sequence

	Description:
	Rauch-Tung-Striebel smoother algorithm. Calculate "smoothed"
	sequence from given Kalman filter output sequence
	by conditioning all steps to all measurements.

	Example:
		m = m0;
		P = P0;
		MM = zeros(size(m,1),size(Y,2));
		PP = zeros(size(m,1),size(m,1),size(Y,2));
		for k=1:size(Y,2)
			[m,P] = kf_predict(m,P,A,Q);
			[m,P] = kf_update(m,P,Y(:,k),H,R);
			MM(:,k) = m;
			PP(:,:,k) = P;
		end
		[SM,SP] = rts_smooth(MM,PP,A,Q);

	See also:
		KF_PREDICT, KF_UPDATE
	"""

	M = M.copy()
	P = P.copy()

	m_1, m_2 = M.shape[:2]

	# Extend A and Q if they are NxN matrices
	if len(set(A.shape)) == 1:
		A = np.tile(A, (m_1, 1, 1))
	if len(set(Q.shape)) == 1:
		Q = np.tile(Q, (m_1, 1, 1))

	# Run the smoother
	D = np.zeros((m_1, m_2, m_2))
	for k in range((m_1-2),-1,-1):
		P_pred   = A[k] @ P[k] @ A[k].T + Q[k]
		D[k] = solve(P_pred.T, A[k]@P[k].T).T
		M[k] = M[k] + D[k] @ (M[k+1] - A[k] @ M[k])
		P[k] = P[k] + D[k] @ (P[k+1] - P_pred) @ D[k].T

	return M, P, D