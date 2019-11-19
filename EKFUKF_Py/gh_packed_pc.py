import numpy as np


def gh_packed_pc(x, fmmparam):
	"""
	GH_PACKED_PC - Pack P and C for the Gauss-Hermite transformation

	Syntax:
		pc = GH_PACKED_PC(x,fmmparam)

	In:
		x - Evaluation point
		fmmparam - Array of handles and parameters to form the functions.

	Out:
		pc - Output values

	Description:
	Packs the integrals that need to be evaluated in nice function form to
	ease the evaluation. Evaluates P = (f-fm)(f-fm)' and C = (x-m)(f-fm)'.

	Copyright (c) 2010 Hartikainen, Särkkä, Solin

	This software is distributed under the GNU General Public
	Licence (version 2 or later); please refer to the file
	Licence.txt, included with the software, for details.
	"""

	f  = fmmparam[0]
	m  = fmmparam[1]
	fm = fmmparam[2]
	if len(fmmparam) >= 4:
			param = fmmparam[3]

	if type(f)==str or callable(f):
		if 'param' not in locals():
			F = f(x)
		else:
			F = f(x, param)
	elif type(f) == np.ndarray:
		F = f@x
	else:
		if 'param' not in locals():
			F = f(x)
		else:
			F = f(x,param)
	d = x.shape[0]
	s = F.shape[0]

	# Compute P = (f-fm)(f-fm)' and C = (x-m)(f-fm)'
	# and form array of [vec(P):vec(C)]
	f_ = F.shape[1]
	pc = np.zeros((s**2+d*s,f_))
	P = np.zeros((s,s))
	C = np.zeros((d,s))
	for k in range(f_):
		for j in range(s):
			for i in range(s):
				P[i,j] = (F[i,k]-fm[i]) * (F[j,k] - fm[j])
			for i in range(d):
				C[i,j] = (x[i,k]-m[i]) * (F[j,k] - fm[j])
		pc[:,k] = np.concatenate([P.reshape(s*s), C.reshape(s*d)])
	return pc
