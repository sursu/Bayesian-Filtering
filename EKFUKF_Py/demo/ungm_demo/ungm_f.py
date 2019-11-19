from numpy import isscalar, array, cos

def ungm_f(x,param=None):
	"""
	State transition function for the UNGM-model.
	
	Copyright (C) 2007 Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	n = param

	x_n = 0.5*x[0] + 25*x[0]/(1+x[0]*x[0]) + 8*cos(1.2*(n-1))

	if x.shape[0] > 1:
		x_n += x[1]

	return x_n[None]
		