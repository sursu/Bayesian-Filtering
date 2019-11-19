from numpy import vstack, zeros, sin, cos


def ekf_sine_dh_dx(x, param=None):
	"""
	Jacobian of the measurement model function in the random sine signal demo

	Copyright (C) 2007 Jouni Hartikainen
	%
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	if len(x.shape) < 3:
		x = x[None]

	f = x[:,0]
	a = x[:,2]

	return vstack([(a*cos(f)).T, zeros((f.shape[0],1)), sin(f).T]).T