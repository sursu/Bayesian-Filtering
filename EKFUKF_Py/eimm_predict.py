import numpy as np

from ekf_predict1 import ekf_predict1

def eimm_predict(X_ip, P_ip, MU_ip, p_ij, ind, dims, A, a, param, Q, nargout=3):
	"""
	IMM_PREDICT  Interacting Multiple Model (IMM) Filter prediction step
	
	Syntax:
	  [X_p,P_p,c_j,X,P] = EIMM_PREDICT(X_ip,P_ip,MU_ip,p_ij,ind,dims,A,a,param,Q)
	
	In:
	  X_ip  - Cell array containing N^j x 1 mean state estimate vector for
	          each model j after update step of previous time step
	  P_ip  - Cell array containing N^j x N^j state covariance matrix for 
	          each model j after update step of previous time step
	  MU_ip - Vector containing the model probabilities at previous time step
	  p_ij  - Model transition matrix
	  ind   - Indices of state components for each model as a cell array
	  dims  - Total number of different state components in the combined system
	  A     - Dynamic model matrices for each linear model and Jacobians of each
	          non-linear model's measurement model function as a cell array
	  a     - Function handles of dynamic model functions for each model
	          as a cell array
	  param - Parameters of a for each model as a cell array
	  Q     - Process noise matrices for each model as a cell array.
	
	Out:
	  X_p  - Predicted state mean for each model as a cell array
	  P_p  - Predicted state covariance for each model as a cell array
	  c_j  - Normalizing factors for mixing probabilities
	  X    - Combined predicted state mean estimate
	  P    - Combined predicted state covariance estimate
	  
	Description:
	  IMM-EKF filter prediction step. If some of the models have linear
	  dynamics standard Kalman filter prediction step is used for those.
	
	See also:
	  EIMM_UPDATE, EIMM_SMOOTH

	History:
	  09.01.2008 JH The first official version.
	
	Copyright (C) 2007,2008 Jouni Hartikainen
	
	$Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

    # Number of models
	m = len(X_ip)
    
    # Construct empty cell arrays for ekf_update if a is not specified
	if a is None:
		a = np.empty(m, dtype=object)
    
    # Same for a's parameters
	if param is None:
		param = np.empty(m, dtype=object)

	# Normalizing factors for mixing probabilities
	c_j = MU_ip@p_ij

	# Mixing probabilities
	MU_ij = p_ij*MU_ip[:,None] / c_j


	# Calculate the mixed state mean for each filter
	X_0j = np.empty(m, dtype=object)
	for j in range(m):
		X_0j[j] = np.zeros((dims,1))
		for i in range(m):
			X_0j[j][ind[i]] += X_ip[i]*MU_ij[i,j]
		

	# Calculate the mixed state covariance for each filter
	P_0j = np.empty(m, dtype=object)
	for j in range(m):
		P_0j[j] = np.zeros((dims,dims))
		for i in range(m):
			P_0j[j][np.ix_(ind[i],ind[i])] += MU_ij[i,j]*(P_ip[i] + (X_ip[i]-X_0j[j][ind[i]])@(X_ip[i]-X_0j[j][ind[i]]).T)


	# Space for predictions
	X_p = np.empty(m, dtype=object)
	P_p = np.empty(m, dtype=object)

	# Make predictions for each model
	for i in range(m):
		X_p[i], P_p[i] = ekf_predict1(X_0j[i][ind[i]], P_0j[i][np.ix_(ind[i],ind[i])], A[i], Q[i], a[i], None, param[i])
		#[X_p{i}, P_p{i}] = kf_predict(X_0j{i}(ind{i}),P_0j{i}(ind{i},ind{i}),A{i},Q{i})


	# Output the combined predicted state mean and covariance, if wanted
	if nargout > 3:
		# Space for estimates
		X = np.zeros((dims,1))
		P = np.zeros((dims,dims))
		
		# Predicted state mean
		for i in range(m):
			X[ind[i]] += MU_ip[i]*X_p[i]
		# Predicted state covariance
		for i in range(m):
			P[np.ix_(ind[i],ind[i])] += MU_ip[i]*(P_p[i] + (X_ip[i]-X[ind[i]])*(X_ip[i]-X[ind[i]]).T)
		
		return X_p, P_p, c_j, X, P

	return X_p, P_p, c_j