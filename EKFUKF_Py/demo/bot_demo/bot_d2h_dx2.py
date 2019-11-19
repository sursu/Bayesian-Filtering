import numpy as np

def bot_d2h_dx2(x, s):
	"""
	Hessian of the measurement function in BOT-demo.

	Copyright (C) 2007 Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	# Space for Hessians. Note that we need a Hessian for
	# each dimension in the measurement space, that is we need
	# a Hessian for each sensor in this case.  

	s_ = s.shape[1]
	x_ = x.shape[0]    

	dY = np.zeros((s_,x_,x_))
	
	# Loop through sensors.
	for i in range(s_):
		# Derivative twice wrt. x
		dx2 = -2*(x[0]-s[0,i]) / ((x[0]-s[0,i])**2+(x[1]-s[1,i])**2)**2
		# Derivative twice wrt. y    
		dy2 = -2*(x[1]-s[1,i]) / ((x[0]-s[0,i])**2+(x[1]-s[1,i])**2)**2
		# Derivative wrt. x and y
		dxdy = ((x[1]-s[1,i])**2-(x[0]-s[0,i])**2) / ((x[0]-s[0,i])**2+(x[1]-s[1,i])**2)**2
		dh = np.array([[dx2, dxdy, 0, 0],
					   [dxdy, dy2, 0, 0],
					   [0,    0,   0, 0],
					   [0,    0,   0, 0]])
		dY[i] = dh
	return dY