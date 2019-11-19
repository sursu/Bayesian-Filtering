from numpy import sin, cos, zeros

def f_turn_dx(x, param):
	"""
	Jacobian of the state transition function in reentry demo.

	
	Copyright (C) 2005-2006 Simo S�rkk�
	              2007      Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	dt  = param
	w = x[4]
	
	if w == 0:
		coswt = 1
		coswto = 0
		coswtopw = 0
		
		sinwt = 0
		sinwtpw = dt
		
		dsinwtpw = 0
		dcoswtopw = -0.5*dt**2
	else:
		coswt = cos(w*dt)
		coswto = cos(w*dt) - 1
		coswtopw = coswto / w
		
		sinwt = sin(w*dt)
		sinwtpw = sinwt/w
		
		dsinwtpw = (w*dt*coswt - sinwt) / (w**2)
		dcoswtopw = (-w*dt*sinwt - coswto) / (w**2)

	df = zeros((5,5))
	
	df[0,0] = 1
	df[0,2] = sinwtpw
	df[0,3] = coswtopw
	df[0,4] = dsinwtpw * x[2] + dcoswtopw * x[3]
	
	df[1,1] = 1
	df[1,2] = -coswtopw
	df[1,3] = sinwtpw
	df[1,4] = -dcoswtopw * x[2] + dsinwtpw * x[3]
	
	df[2,2] = coswt
	df[2,3] = -sinwt
	df[2,4] = -dt * sinwt * x[2] - dt * coswt * x[3]
	
	df[3,2] = sinwt
	df[3,3] = coswt
	df[3,4] = dt * coswt * x[2] - dt * sinwt * x[3]
	
	df[4,4] = 1

	return df
