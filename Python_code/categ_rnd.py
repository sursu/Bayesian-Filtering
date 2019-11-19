import numpy as np


def categ_rnd(P, N=1):
	"""
	CATEG_RND  Draws samples from a given one dimensional discrete distribution
	
	Syntax:
	  C = CATEG_RND(P,N)
	
	In:
	  P - Discrete distribution, which can be a numeric array
	      of probabilities or a cell array of particle structures,
	      whose weights represent the distribution.
	  N - Number of samples (optional, default 1)
	
	Out:
	  C - Samples in a Nx1 vector
	
	Description:
	  Draw random category

	Copyright (C) 2002 Simo S�rkk�
	              2008 Jouni Hartikainen
	
	$Date: 2013/08/26 12:58:41 $
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""
	
	# If particle structures are given
	if type(P)==object: # and isfield([P],'W'):
		tmp = [P]
		P = [tmp[W]]
	
	# Draw the categories
	C = np.zeros(N, dtype=int)
	P /= np.sum(P)
	P = np.cumsum(P)
	for i in range(N):
		C[i] = np.argmax(P > np.random.uniform())
	
	return C