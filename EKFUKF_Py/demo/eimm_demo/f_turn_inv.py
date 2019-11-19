from numpy import zeros, array, sin, cos
from numpy.linalg import solve

def f_turn_inv(x,param):
	"""
	% Inverse prediction for the coordinated turn model
	% used in extended IMM filter demonstration
	%
	% 

	% Copyright (C) 2007 Jouni Hartikainen
	%
	% This software is distributed under the GNU General Public
	% Licence (version 2 or later); please refer to the file
	% Licence.txt, included with the software, for details.
	"""
	
	dt = param

	Y = zeros(x.shape)

	for i in range(x.shape[1]):
		w = x[4,i]
		
		if w == 0:
			coswt = cos(w*dt)
			coswto = cos(w*dt)-1
			coswtopw = 0
			
			sinwt = 0
			sinwtpw = dt
		else:
			coswt = cos(w*dt)
			coswto = cos(w*dt)-1
			coswtopw = coswto/w
			
			sinwt = sin(w*dt)
			sinwtpw = sinwt/w

		F = array([[1, 0, sinwtpw,   coswtopw,  0],
				[0, 1,-coswtopw,  sinwtpw,   0],
				[0, 0, coswt,    -sinwt,     0],
				[0, 0, sinwt,     coswt,     0],
				[0, 0, 0,         0,         1]])

		Y[:,i] = solve(F,x[:,i])

	return Y