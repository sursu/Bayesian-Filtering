import numpy as np

from ekf_update1 import ekf_update1

def eimm_update(X_p, P_p, c_j, ind, dims, Y, H, h, R, param, nargout=5):
	"""	
	IMM_UPDATE  Interacting Multiple Model (IMM) Filter update step
	
	Syntax:
	  [X_i,P_i,MU,X,P] = IMM_UPDATE(X_p,P_p,c_j,ind,dims,Y,H,h,R,param)
	
	In:
	  X_p  - Cell array containing N^j x 1 mean state estimate vector for
	         each model j after prediction step
	  P_p  - Cell array containing N^j x N^j state covariance matrix for 
	         each model j after prediction step
	  c_j  - Normalizing factors for mixing probabilities
	  ind  - Indices of state components for each model as a cell array
	  dims - Total number of different state components in the combined system
	  Y    - Dx1 measurement vector.
	  H    - Measurement matrices for each linear model and Jacobians of each
	         non-linear model's measurement model function as a cell array
	  h    - Cell array containing function handles for measurement functions
	         for each model having non-linear measurements
	  R    - Measurement noise covariances for each model as a cell array.
	  param - Parameters of h
	
	Out:
	  X_i  - Updated state mean estimate for each model as a cell array
	  P_i  - Updated state covariance estimate for each model as a cell array
	  MU   - Estimated probabilities of each model
	  X    - Combined updated state mean estimate
	  P    - Combined updated covariance estimate
	  
	Description:
	  IMM-EKF filter measurement update step. If some of the models have linear
	  measurements standard Kalman filter update step is used for those.
	
	See also:
	  IMM_PREDICT, IMM_SMOOTH, IMM_FILTER

	History:
	  01.11.2007 JH The first official version.
	
	Copyright (C) 2007 Jouni Hartikainen
	
	$Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	# Number of models 
	m = len(X_p)

	# Construct empty cell arrays for ekf_update if h is not specified
	if h is None:
		h = np.empty(m, dtype=object)
	
	# Same for h's parameters
	if param is None:
		param = np.empty(m, dtype=object)
	
	
	# Space for update state mean, covariance and likelihood of measurements
	X_i = np.empty(m, dtype=object)
	P_i = np.empty(m, dtype=object)
	lbda = np.zeros(m)

	# Update for each model
	for i in range(m):
		# Update the state estimates
		#[X_i{i}, P_i{i}, K, IM, IS, lbda(i)] = ekf_update1(X_p{i},P_p{i},Y,H{i},R{i},h{i},[],param{i})
		X_i[i], P_i[i], _, _, _, lbda[i] = ekf_update1(X_p[i], P_p[i], Y, H[i], R[i], h[i], None, param[i], nargout=6)
		#[X_i{i}, P_i{i}, K, IM, IS] = kf_update(X_p{i},P_p{i},Y,H{i},R{i})

		# Calculate measurement likelihoods for each model
		#lbda(i) = kf_lhood(X_p{i},P_p{i},Y,H{i},R{i})
	
	
	# Calculate the model probabilities
	MU = np.zeros(m) 
	c = np.sum(lbda*c_j)
	MU = c_j*lbda/c
	
	# In case lbda's happen to be zero    
	# if c == 0
	if c == 0:
		MU = c_j
		#X_i = X_p
		#P_i = P_p
	
	
	# Output the combined updated state mean and covariance, if wanted.
	if nargout > 3:
		# Space for estimates
		X = np.zeros((dims,1))
		P = np.zeros((dims,dims))
		# Updated state mean
		for i in range(m):
			X[ind[i]] += MU[i]*X_i[i]
		
		# Updated state covariance
		for i in range(m):
			P[np.ix_(ind[i],ind[i])] += MU[i]*(P_i[i] + (X_i[i]-X[ind[i]])@(X_i[i]-X[ind[i]]).T)
		
	

	return X_i, P_i, MU, X, P
	
	