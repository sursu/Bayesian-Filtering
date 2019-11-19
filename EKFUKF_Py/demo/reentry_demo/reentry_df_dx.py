from numpy import sqrt, exp, eye, zeros, float128

def reentry_df_dx(x,param):
	"""
	Jacobian of the state transition function in reentry demo.

	
	Copyright (C) 2005-2006 Simo S�rkk�
	              2007      Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	dt, b0, H0, Gm0, R0  = param

	R = sqrt(x[0]**2 + x[1]**2)
	V = sqrt(x[2]**2 + x[3]**2)
	b = b0 * exp(x[4])
	D = b * exp((R0-R)/H0) * V
	G = -Gm0 / R**3

	dR_dx1 = x[0] / R
	dR_dx2 = x[1] / R
	dV_dx3 = x[2] / V
	dV_dx4 = x[3] / V
	db_dx5 = b
	dD_dx1 = b * (-dR_dx1/H0) * exp((R0-R)/H0) * V
	dD_dx2 = b * (-dR_dx2/H0) * exp((R0-R)/H0) * V
	dD_dx3 = b * exp((R0-R)/H0) * dV_dx3
	dD_dx4 = b * exp((R0-R)/H0) * dV_dx4
	dD_dx5 = db_dx5 * exp((R0-R)/H0) * V
	dG_dx1 = -Gm0 * (-3 * dR_dx1 / R**4)
	dG_dx2 = -Gm0 * (-3 * dR_dx2 / R**4)

	df = zeros((5,5))
	df[0,2] = 1
	df[1,3] = 1
	df[2,0] = dD_dx1 * x[2] + dG_dx1 * x[0] + G
	df[2,1] = dD_dx2 * x[2] + dG_dx2 * x[0]
	df[2,2] = dD_dx3 * x[2] + D
	df[2,3] = dD_dx4 * x[2]
	df[2,4] = dD_dx5 * x[2]
	
	df[3,0] = dD_dx1 * x[3] + dG_dx1 * x[1] 
	df[3,1] = dD_dx2 * x[3] + dG_dx2 * x[1] + G
	df[3,2] = dD_dx3 * x[3]
	df[3,3] = dD_dx4 * x[3] + D
	df[3,4] = dD_dx5 * x[3]

	da = eye(df.shape[0]) + dt * df

	return da
	
