import numpy as np
from numpy.linalg import inv, solve

from kf_predict import kf_predict
from kf_update import kf_update
from gauss_pdf import gauss_pdf

def imm_smooth(MM, PP, MM_i, PP_i, MU, p_ij, mu_0j, ind, dims, A, Q, R, H, Y):
	"""
	IMM_SMOOTH   Fixed-interval IMM smoother using two IMM-filters.
	
	Syntax:
	  [X_S,P_S,X_IS,P_IS,MU_S] = IMM_SMOOTH(MM,PP,MM_i,PP_i,MU,p_ij,mu_0j,ind,dims,A,Q,R,H,Y)
	
	In:
	  MM    - NxK matrix containing the means of forward-time 
	          IMM-filter on each time step
	  PP    - NxNxK matrix containing the covariances of forward-time
	          IMM-filter on each time step
	  MM_i  - Model-conditional means of forward-time IMM-filter on each time step
	          as a cell array
	  PP_i  - Model-conditional covariances of forward-time IMM-filter on each time
	          step as a cell array
	  MU    - Model probabilities of forward-time IMM-filter on each time step 
	  p_ij  - Model transition probability matrix
	  mu_0j - Prior model probabilities
	  ind   - Indices of state components for each model as a cell array
	  dims  - Total number of different state components in the combined system
	  A     - State transition matrices for each model as a cell array.
	  Q     - Process noise matrices for each model as a cell array.
	  R     - Measurement noise matrices for each model as a cell array.
	  H     - Measurement matrices for each model as a cell array
	  Y     - Measurement sequence
	
	
	Out:
	  X_S  - Smoothed state means for each time step
	  P_S  - Smoothed state covariances for each time step
	  X_IS - Model-conditioned smoothed state means for each time step
	  P_IS - Model-conditioned smoothed state covariances for each time step
	  MU_S - Smoothed model probabilities for each time step
	  
	Description:
	  Two filter fixed-interval IMM smoother.
	
	See also:
	  IMM_UPDATE, IMM_PREDICTION, IMM_FILTER

	History:
	  01.11.2007 JH The first official version.
	
	Copyright (C) 2007 Jouni Hartikainen
	
	$Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""


	# Default values for mean and covariance
	MM_def = np.zeros((dims,1))
	PP_def = np.diag(np.ones(dims))

	# Number of models
	m = len(A)
	
	# Number of measurements
	n = len(Y)

	# The prior model probabilities for each step
	p_jk = np.zeros((n,m))
	p_jk[0] = mu_0j
	for i1 in range(1,n):
		for i2 in range(m):
			p_jk[i1,i2] = np.sum(p_ij[i2]*p_jk[i1-1])
	
	# Backward-time transition probabilities
	p_ijb = np.zeros((n,m,m))
	for k in range(n):
		for i1 in range(m):
			# Normalizing constant
			b_i = np.sum(p_ij[i1]*p_jk[k])
			for j in range(m):
				p_ijb[k][i1,j] = p_ij[j,i1]*p_jk[k,j] / b_i

	
	# Space for overall smoothed estimates
	x_sk = np.zeros((n,dims,1))
	P_sk = np.zeros((n,dims,dims))
	mu_sk = np.zeros((n,m))
	
	# Values of smoothed estimates at the last time step.
	x_sk[-1] = MM[-1]
	P_sk[-1] = PP[-1]
	mu_sk[-1] = MU[-1]
	
	# Space for model-conditioned smoothed estimates
	x_sik = np.empty((n,m), dtype=object)
	P_sik = np.empty((n,m), dtype=object)
	
	# Values for last time step
	x_sik[-1] = MM_i[-1]
	P_sik[-1] = PP_i[-1]
	
	# Backward-time estimated model probabilities
	mu_bp = MU[-1]
	
	# Space for model-conditioned backward-time updated means and covariances
	x_bki = MM_i[-1]
	P_bki = PP_i[-1]
	
	# Space for model-conditioned backward-time predicted means and covariances
	x_kp = np.tile(MM_def, (m,1,1))
	P_kp = np.tile(PP_def, (m,1,1))
	
	for k in range(n-2,-1,-1):
		# Space for normalizing constants and conditional model probabilities
		a_j = np.zeros(m)
		mu_bijp = np.zeros((m,m))
		
		for i2 in range(m):
			# Normalizing constant
			a_j[i2] = np.sum(p_ijb[k][i2]*mu_bp)
			# Conditional model probability
			mu_bijp[:,i2] = p_ijb[k][i2]*mu_bp / a_j[i2]

			# Backward-time KF prediction step
			x_kp[i2][ind[i2]], P_kp[i2][np.ix_(ind[i2],ind[i2])] = kf_predict(x_bki[i2], P_bki[i2], inv(A[i2]), Q[i2])
		
		# Space for mixed predicted mean and covariance
		x_kp0 = np.tile(MM_def, (m,1,1))
		P_kp0 = np.tile(PP_def, (m,1,1))
		
		# Space for measurement likelihoods
		lhood_j = np.zeros(m)

		for i2 in range(m):
			# Initialize with default values         
			P_kp0[i2][np.ix_(ind[i2],ind[i2])] = np.zeros((len(ind[i2]),len(ind[i2])))
			
			# Mix the mean
			for i1 in range(m):
				x_kp0[i2][ind[i2]] += mu_bijp[i1,i2]*x_kp[i1][ind[i2]]
			
			# Mix the covariance 
			for i1 in range(m):
				P_kp0[i2][np.ix_(ind[i2],ind[i2])] += mu_bijp[i1,i2]*(P_kp[i1][np.ix_(ind[i2],ind[i2])] + (x_kp[i1][ind[i2]] - x_kp0[i2][ind[i2]])@(x_kp[i1][ind[i2]]-x_kp0[i2][ind[i2]]).T) 
			

			# Backward-time KF update
			x_bki[i2][ind[i2]], P_bki[i2][np.ix_(ind[i2],ind[i2])], _, _, _, lhood_j[i2] = kf_update(x_kp0[i2][ind[i2]], P_kp0[i2][np.ix_(ind[i2],ind[i2])],Y[k], H[i2], R[i2])
		
		# Normalizing constant
		a = lhood_j @ a_j
		# Updated model probabilities
		mu_bp = a_j*lhood_j / a
		
		# Space for conditional measurement likelihoods
		lhood_ji = np.zeros((m,m))
		for i1 in range(m):
			for i2 in range(m):
				d_ijk = MM_def.copy()
				D_ijk = PP_def.copy()
				d_ijk += x_kp[i1]
				d_ijk[ind[i2]] -= MM_i[k,i2]
				PP2 = np.zeros((dims,dims))
				PP2[np.ix_(ind[i2],ind[i2])] = PP_i[k,i2]
				D_ijk = P_kp[i1] + PP2

				# Calculate the (approximate) conditional measurement likelihoods
				#D_ijk = 0.01^2*eye(size(D_ijk))
				lhood_ji[i2,i1], _ = gauss_pdf(d_ijk, 0, D_ijk)               
			
		
		
		d_j = np.zeros(m)
		for i2 in range(m):
			d_j[i2] = p_ij[i2] @ lhood_ji[i2]
		
		d = d_j @ MU[k]
		
		mu_ijsp = np.zeros((m,m))
		for i1 in range(m):
			for i2 in range(m):
				mu_ijsp[i1,i2] = p_ij[i2,i1]*lhood_ji[i2,i1] / d_j[i2]
			
		
				
		mu_sk[k] = d_j*MU[k] / d

		# Space for two-step conditional smoothing distributions p(x_k^j|m_{k+1}^i,y_{1:N}),
		# which are a products of two Gaussians
		x_jis = np.empty((m,m), dtype=object)
		P_jis = np.empty((m,m), dtype=object)
		for i2 in range(m):
			for i1 in range(m):
				MM1 = MM_def.copy()
				MM1[ind[i2]] = MM_i[k,i2]
				
				PP1 = PP_def.copy()
				PP1[np.ix_(ind[i2],ind[i2])] = PP_i[k,i2]

				# iPP1 = inv(PP1)
				# iPP2 = inv(P_kp[i1])
				
				# Covariance of the Gaussian product
				#P_jis{i2,i1} = inv(iPP1+iPP2)
				P_jis[i2,i1] = solve((PP1+P_kp[i1]), PP1).T @ P_kp[i1]
				# Mean of the Gaussian product
				x_jis[i2,i1] = P_jis[i2,i1] @ (solve(PP1,MM1) + solve(PP2,x_kp[i1]))
			
		

		# Mix the two-step conditional distributions to yield model-conditioned
		# smoothing distributions.
		for i2 in range(m):
			# Initialize with default values
			x_sik[k,i2] = MM_def.copy()
			P_sik[k,i2] = PP_def.copy()
			P_sik[k,i2][np.ix_(ind[i2],ind[i2])] = np.zeros((len(ind[i2]),len(ind[i2])))
			
			# Mixed mean
			for i1 in range(m):
				x_sik[k,i2] += mu_ijsp[i1,i2]*x_jis[i2,i1]
			
			# Mixed covariance
			for i1 in range(m):
				P_sik[k,i2] += mu_ijsp[i1,i2]*(P_jis[i2,i1] + (x_jis[i2,i1]-x_sik[k,i2])@(x_jis[i2,i1]-x_sik[k,i2]).T) 
		

		# Mix the overall smoothed mean
		for i1 in range(m):
			x_sk[k] += mu_sk[k,i1] * x_sik[k,i1]
		
		# Mix the overall smoothed covariance
		for i1 in range(m):
			P_sk[k] += mu_sk[k,i1]*(P_sik[k,i1] + (x_sik[k,i1]-x_sk[k])@(x_sik[k,i1]-x_sk[k]).T)
		
	
	return x_sk, P_sk, x_sik, P_sik, mu_sk