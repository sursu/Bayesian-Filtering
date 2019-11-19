from reentry_f import reentry_f

def reentry_if(x, param):
	"""
	% Inverse dynamical function for reentry demo.

	% 
	% Copyright (C) 2005-2006 Simo S�rkk�
	%
	% This software is distributed under the GNU General Public 
	% Licence (version 2 or later); please refer to the file 
	% Licence.txt, included with the software, for details.
	"""

	y = reentry_f(x,param)
	x = 2 * x[:y.shape[0]] - y

	return x