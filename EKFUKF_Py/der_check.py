import numpy as np

def der_check(f=None, df=None, index=None, *args, nargout=1):
	"""
	DER_CHECK  Check derivatives using finite differences
	
	Syntax:
	  [D0,D1] = DER_CHECK(F,DF,INDEX,[P1,P2,P3,...])
	
	In:
	  F  - Name of actual function or inline function
	       in form F(P1,P2,...)
	  DF - Derivative value as matrix, name of derivative
	       function or inline function in form DF(P1,P2,...).
	  INDEX - Index of parameter of interest. DF should
	       Calculate the derivative with recpect to parameter
	       Pn, where n is the index.
	
	Out:
	  D0 - Actual derivative
	  D1 - Estimated derivative
	
	Description:
	  Evaluates function derivative analytically and
	  using finite differences. If no output arguments
	  are given, issues a warning if these two values
	  differ too much.
	
	  Function is intended to checking that derivatives
	  of transition and measurement equations of EKF are
	  bug free.
	
	See also:
	  EKF_PREDICT1, EKF_UPDATE1, EKF_PREDICT2, EKF_UPDATE2

	History:
	  12.03.2003  SS  Support for function handles
	  27.11.2002  SS  The first official version.
	
	Copyright (C) 2002 Simo S�rkk�
	
	$Id$
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
	"""

	# Calculate function value and derivative
	if type(f)==str or callable(f):
		y0 = f(*args)
	else:
		y0 = f(*args)

	if type(df) == np.ndarray:
		D0 = df
	elif type(df)==str or callable(df):
		D0 = df(*args)
	else:
		D0 = df(*args)


	# Calculate numerical derivative
	h = 0.0000001
	X = args[index-1]
	args = list(args)
	D1 = np.zeros(X.shape)
	for r in range(X.shape[0]):
		for c in range(X.shape[1]):
			H = np.zeros(X.shape)
			H[r,c] = h
			args[index-1] = X+H
			if type(f)==str or callable(f):
				y1 = f(*args)
			else:
				y1 = f(*args)

			d1 = (y1-y0)/h
			if d1.shape[0]>1:
				D1[r] = d1
			else:
				D1[r,c] = d1

	if nargout==0:
		d = np.abs(D1-D0.T)
		if np.max(d) > 0.001:
			print('Derivatives differ too much')
			print(np.max(d))
		else:
			print('Derivative check passed.\n')
	return D0, D1

