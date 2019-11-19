#################################################################
#
# Simulate pendulum data for the examples in the book
#
# Simo Sarkka (2013), Bayesian Filtering and Smoothing,
# Cambridge University Press. 
#
# Last updated: $Date: 2013/08/26 12:58:41 $.
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.
#
#################################################################

import numpy as np
from numpy.linalg import cholesky

import plotly.graph_objects as go

##
# Simulate simple pendulum. Note that the system easily
# diverges, but it should not matter.

dt = 0.01
g  = 9.81
Q  = 0.01*np.array([[dt**3/3, dt**2/2],
					[dt**2/2,      dt]])
R  = 0.1
m0 = np.array([1.6, 0])[:,None] # Slightly off
P0 = 0.1*np.eye(2)
	
steps = 500

QL = cholesky(Q)

T = np.arange(dt, steps*dt+dt, dt)
X = np.zeros((steps,m0.shape[0],1))
Y = np.zeros((steps,1,1))
x = np.array([1.5, 0])[:,None]
for k in range(steps):
	x = np.array([x[0]+x[1]*dt, x[1]-g*np.sin(x[0])*dt])
	w = QL @ np.random.normal(size=(2,1))
	x += w
	y = np.sin(x[0]) + np.sqrt(R)*np.random.normal()
	X[k] = x
	Y[k] = y

# Plot the data
fig = go.Figure()
fig.add_scatter(x=T, y=X[:,0,0])
fig.add_scatter(x=T, y=Y[:,0,0], mode='markers')
fig.layout.update(height=600, showlegend=False)
fig.show()

