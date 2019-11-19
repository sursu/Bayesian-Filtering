from numpy import sqrt, exp, zeros


def reentry_f(xw,param):
	"""
	Dynamical model function for reentry problem.
	Discretization is done using a simple Euler
	time integration.

	
	Copyright (C) 2005-2006 Simo S�rkk�
	              2007      Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	dt, b0, H0, Gm0, R0 = param[:5]
	
	x = xw[:5].copy()

	R = sqrt(x[0]**2 + x[1]**2)
	V = sqrt(x[2]**2 + x[3]**2)
	b = b0 * exp(x[4])
	D = b * exp((R0-R)/H0) * V
	G = -Gm0 / R**3
	dot_x = zeros(x.shape)
	dot_x[0] = x[2]
	dot_x[1] = x[3]
	dot_x[2] = D * x[2] + G * x[0]
	dot_x[3] = D * x[3] + G * x[1]
	dot_x[4] = zeros((1,x.shape[1]))

	# Euler integration
	x += dt * dot_x

	# Add process noise if the state is augmented 
	if xw.shape[0] > 5 and len(param) > 5:
		L = param[5]
		w = xw[L.shape[0] : L.shape[0]+L.shape[1]]
		x += L @ w

	return x
