import numpy as np

def resampstr(p):
	"""
	RESAMPSTR Stratified resampling
	  S = RESAMPSTR(P) returns a new set of indices according to 
	  the probabilities P. P is array of probabilities, which are
	  not necessarily normalized, though they must be non-negative,
	  and not all zero. The size of S is the size of P.
	
	  Default is to use no-sort resampling. For sorted resampling use
	   [PS,PI]=SORT(P)
	   S=PI(RESAMPSTR(PS))
	  Sorted re-sampling is slower but has slightly smaller variance.
	  Stratified resampling is unbiased, almost as fast as
	  deterministic resampling (RESAMPDET), and has only slightly
	  larger variance.
	
	  In stratified resampling indices are sampled using random
	  numbers u_j~U[(j-1)/n,j/n], where n is length of P. Compare
	  this to simple random resampling where u_j~U[0,1]. See,
	  Kitagawa, G., Monte Carlo Filter and Smoother for
	  Non-Gaussian Nonlinear State Space Models, Journal of
	  Computational and Graphical Statistics, 5(1):1-25, 1996. 
	
	  See also RESAMPSIM, RESAMPRES, RESAMPDET

	Copyright (c) 2003-2004 Aki Vehtari
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""
	m, n = np.matrix(p).shape

	mn = m*n
	pn = p/np.sum(p)*mn

	s = np.zeros(n, dtype=int)
	r = np.random.uniform(size=mn)

	k = 0
	c = 0
	for i in range(mn):
		c += pn[i]
		if c >= 1:
			a = int(c)
			c -= a
			s[k:a+k] = i
			k += a
		if (k < mn) and (c >= r[k]):
			c -= 1
			k += 1
			s[k-1] = i
	return s