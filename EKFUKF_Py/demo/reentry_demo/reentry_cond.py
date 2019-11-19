#
# Compute condition numbers of transition matrices
# for re-entry simulation with no noise.
#

# Copyright (C) 2005-2006 Simo S�rkk�
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.


from numpy import zeros

print('[Generating data...]\n')

nsteps = round(200/dt)
x = true_m0
X = zeros(x.shape[0],nsteps)
Y = zeros(2,nsteps)
T = zeros(1,nsteps)

C = zeros(1,nsteps)
t = 0
for k in range(nsteps):
	ddt = dt / sim_iter
	for i in range(sim_iter):
		A = reentry_df_dx(x, [ddt, b0, H0, Gm0, R0])
		x = reentry_f(x, [ddt, b0, H0, Gm0, R0])

	c = cond(A)
	
	t += dt
	C[k] = c
	T[k] = t