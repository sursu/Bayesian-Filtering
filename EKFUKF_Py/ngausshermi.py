import numpy as np
from numpy.linalg import cholesky

from hermitepolynomial import hermitepolynomial

def ngausshermi(f, n, m, P, param=None):
	"""
	NGAUSSHERMI - ND scaled Gauss-Hermite quadrature (cubature) rule
	
	Syntax:
		[I,x,W,F] = ngausshermi(f,p,m,P,param)
	
	In:
		f - Function f(x,param) as inline, name or reference
		n - Polynomial order
		m - Mean of the d-dimensional Gaussian distribution
		P - Covariance of the Gaussian distribution
		param - Optional parameters for the function
	
	Out:
		I - The integral value
		x - Evaluation points
		W - Weights
		F - Function values
	
	Description:
		Approximates a Gaussian integral using the Gauss-Hermite method
		in multiple dimensions:
			int f(x) N(x | m,P) dx

	History:
		2009 - Initial version (ssarkka)
		2010 - Partially rewritten (asolin)

	Copyright (c) 2010 Simo Sarkka, Arno Solin
			This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.

		THEORY OF OPERATION:
	
	Consider the multidimensional integral:
	
		int f(x1,...,xn) exp(-x1^2-...-xn^2) dx1...dxn
	
	If we form quadrature for dimension 1, then
	for dimension 2 and so on, we get
	
	= sum w_i1 int f(x1^i1,...,xn) exp(-x2^2-...-xn^2) dx2...dxn
	= sum w_i1 w_i1 int f(x1^i1,x2^i2,...,xn) exp(-x3^2-...-xn^2) dx3...dxn
	= ...
	= sum w_i1 ... w_in f(x1^i1,...,xn^in) 
	
	Now let X = [x1,...,xn] and consider
	
		1/(2pi)^{n/2}/sqrt(|P|) int f(X) exp(-1/2 (X-M)' iP (X-M)) dX
	
	Let P = L L' and change variable to
	
		X = sqrt(2) L Y + M  <=>  Y = 1/sqrt(2) iL (X-M)
	
	then dX = sqrt(2)^n |L] dY = sqrt(2)^n sqrt(|P]) dY i.e. we get
	
		1/(2pi)^{n/2}/sqrt(|P|) int f(X) exp(-1/2 (X-M)' iP (X-M)) dX
		= 1/(pi)^{n/2} int f(sqrt(2) L Y + M) exp(-Y'Y) dY
		= int g(Y) exp(-Y'Y) dY
	
	which is of the previous form if we define
	
		g(Y) = 1/(pi)^{n/2} f(sqrt(2) L Y + M)
	"""


	# The Gauss-Hermite cubature rule

	# The hermite polynomial of order n
	p = hermitepolynomial(n)
	
	# Evaluation points
	x = np.roots(p)
	
	# 1D coefficients
	Wc = 2**(n-1) * np.math.factorial(n) * np.sqrt(np.pi)/n**2
	p2 = hermitepolynomial(n-1)
	W  = np.zeros(n)
	for i in range(n):
		W[i] = Wc * np.polyval(p2, x[i])**(-2)
			
	d = m.shape[0]
	if d == 1:
		x = x.T
	
	# Generate all n^d collections of indexes by
	# transforming numbers 0...n^d-1) into n-base system
	num = np.arange(n**d)
	ind = np.zeros((d, n**d), dtype=int)
	for i in range(d):
		ind[i] = num%n
		num //= n
	
	# Form the sigma points and weights
	L = cholesky(P)
	SX = np.sqrt(2)*L@x[ind] + np.tile(m, ind.shape[1])
	W = np.prod(W[ind], axis=0)              # ND weights
	
	# Evaluate the function at the sigma points
	if type(f)==str or callable(f):
		if param is None:
			F = f(SX)
		else:
			F = f(SX,param)
	elif type(f) == np.ndarray:
		F = f@SX
	else:
		if param is None:
			F = f(SX)
		else:
			F = f(SX,param)

	# Evaluate the integral
	I = np.sum(F*np.tile(W,(F.shape[0],1)), axis=1, keepdims=True) / np.sqrt(np.pi)**d

	return I, SX, x, W, F