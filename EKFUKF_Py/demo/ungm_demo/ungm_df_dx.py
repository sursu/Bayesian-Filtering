
def ungm_df_dx(x,param):
	"""
	% Jacobian of the state transition function for the UNGM-model.
	%
	% Copyright (C) 2007 Jouni Hartikainen
	%
	% This software is distributed under the GNU General Public 
	% Licence (version 2 or later); please refer to the file 
	% Licence.txt, included with the software, for details.
	"""

	df = 0.5+25*(1-x**2) / (1+x**2)**2

	return df
	