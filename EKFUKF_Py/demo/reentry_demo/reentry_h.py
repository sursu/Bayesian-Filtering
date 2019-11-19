from numpy import zeros, sqrt, arctan2

def reentry_h(x, param):
	"""
	Measurement model function for reentry demo.
	
	Copyright (C) 2005-2006 Simo S�rkk�
	              2007      Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	xr, yr = param
	y = zeros((2,x.shape[1]))
	y[0] = sqrt((x[0]-xr)**2 + (x[1]-yr)**2)
	y[1] = arctan2(x[1]-yr, x[0]-xr)
	
	if x.shape[0] == 10:
		y[0] += x[8]
		y[1] += x[9]
	elif x.shape[0] == 7:
		y[0] += x[5]
		y[1] += x[6]
	  
	return y