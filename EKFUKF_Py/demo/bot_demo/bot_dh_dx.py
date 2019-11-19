import numpy as np

def bot_dh_dx(x,s):
	"""
	Jacobian of the measurement model function in BOT demo.
	
	 dh_dx = -(y-sy) / (x-sx)^2 * 1 / (1 + (y-sy)^2 / (x-sx)^2)
	       = -(y-sy) / ((x-sx)^2 + (y-sy)^2)
	 dh_dy = 1 / (x-sx) * 1 / (1 + (y-sy)^2 / (x-sx)^2)
	       = (x-sx) / ((x-sx)^2 + (y-sy)^2)
	

	Copyright (C) 2003 Simo S�rkk�
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	x_ = x.shape[0]
	s_ = s.shape[1]

	# Reserve space for the Jacobian. 
	dY = np.zeros((s_,x_))

	# Loop through sensors
	for i in range(s_):
		dh = np.concatenate([-(x[1]-s[1,i]) / ((x[0]-s[0,i])**2 + (x[1]-s[1,i])**2), (x[0]-s[0,i]) / ((x[0]-s[0,i])**2 + (x[1]-s[1,i])**2), np.zeros(x_-2)])
		dY[i] = dh

	return dY