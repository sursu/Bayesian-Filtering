from numpy import sin

def ekf_sine_h(x, param=None):
	"""
	Measurement model function for the random sine signal demo

	Copyright (C) 2007 Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	if len(x.shape) < 3:
		if x.shape[1] == 1:
			x = x[None]
		else:
			x = x[None].T

	f = x[:,0]
	a = x[:,2]

	Y = a*sin(f)
	if x.shape[1] == 7:
		Y += x[:,6]

	return Y.T
