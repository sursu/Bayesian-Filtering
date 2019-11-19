from numpy import zeros, array, sin, cos

def ekf_sine_d2h_dx2(x,param):
	"""
	Hessian of the measurement model function in the random sine signal demo

	Copyright (C) 2007 Jouni Hartikainen

	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	f = x[0,0]
	a = x[2,0]

	df = zeros((1,3,3))
	df[0] = array([[-a*sin(f), 0, cos(f)],
				   [        0, 0,      0],
				   [   cos(f), 0,      0]])
	return df
