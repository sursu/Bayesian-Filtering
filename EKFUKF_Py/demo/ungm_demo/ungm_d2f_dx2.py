
def ungm_d2f_dx2(x,param):
	"""
	Hessian of the state transition function in UNGM-model.

	Copyright (C) 2007 Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	d = 25*(-6*x+2*x**5-4*x**3) / (1+x**2)**4

	return d