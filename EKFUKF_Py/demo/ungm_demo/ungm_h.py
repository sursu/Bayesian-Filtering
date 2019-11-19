
def ungm_h(x_n, param=None):
	"""
	% Measurement model function for the UNGM-model.
	%
	% Copyright (C) 2007 Jouni Hartikainen
	%
	% This software is distributed under the GNU General Public 
	% Licence (version 2 or later); please refer to the file 
	% Licence.txt, included with the software, for details.
	"""

	# if len(x_n.shape) == 1:
	# 	x_n = x_n[:,None]

	y_n = x_n[0]*x_n[0] / 20
	if x_n.shape[0] == 3:
		y_n += x_n[2]

	return y_n[None]