from numpy import zeros, sqrt, arctan2

def reentry_dh_dx(x, param):
	"""
	Jacobian of the measurement model function in reentry demo.

	
	Copyright (C) 2005-2006 Simo S�rkk�
	              2007      Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	xr, yr = param

	y = zeros((2, x.shape[1]))
	y[0] = sqrt((x[0]-xr)**2 + (x[1]-yr)**2)
	y[1] = arctan2(x[1]-yr, x[0]-xr)
	
	dy = zeros((2, x.shape[0]))

	dy[0,0] = (x[0]-xr) / y[0]
	dy[0,1] = (x[1]-yr) / y[0]
	dy[1,0] = -(x[1]-yr) / ((x[0]-xr)**2 + (x[1]-yr)**2)
	dy[1,1] = (x[0]-xr) / ((x[0]-xr)**2 + (x[1]-yr)**2)

	return dy