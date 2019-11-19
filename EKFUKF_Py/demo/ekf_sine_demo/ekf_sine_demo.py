# Demonstration for EKF using a random sine signal model. 
#
# A Very simple demonstration for extended Kalman filter (EKF), which is
# used to track a random single-component sinusoid signal,
# which is modelled as x_k = a_k*sin(\theta_k), dtheta/dt = omega_k.
# The signal is also filtered with unscented Kalman filter (UKF) for
# comparison.
#
# Copyright (C) 2007 Jouni Hartikainen
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.


import numpy as np

import plotly.graph_objects as go
from ipywidgets import interact

import os, sys
sys.path.append('../..')

from ekf_sine_f import ekf_sine_f
from ekf_sine_h import ekf_sine_h
from ekf_sine_dh_dx import ekf_sine_dh_dx
from ekf_sine_d2h_dx2 import ekf_sine_d2h_dx2

from der_check import der_check
from lti_disc import lti_disc
from gauss_rnd import gauss_rnd

from ekf_predict1 import ekf_predict1
from ekf_update1 import ekf_update1
from ekf_update2 import ekf_update2
from erts_smooth1 import erts_smooth1
from etf_smooth1 import etf_smooth1
from ukf_predict3 import ukf_predict3
from ukf_update3 import ukf_update3
from urts_smooth1 import urts_smooth1

'Filtering the signal with EKF...'

save_plots = 1

# Measurement model and it's derivative
f_func = ekf_sine_f
h_func = ekf_sine_h
dh_dx_func = ekf_sine_dh_dx
d2h_dx2_func = ekf_sine_d2h_dx2

# Initial values for the signal.
f = 0
w = 10
a = 1

# Number of samples and stepsize.
d = 5
n = 500
dt = d/n
x = np.arange(1,n+1)

# Check the derivative of the measurement function.
der_check(h_func, dh_dx_func, 1, np.array([f, w, a])[:,None])

# Dynamic state transition matrix in continous-time domain.
F = np.array([[0, 1, 0],
			  [0, 0, 0],
			  [0, 0, 0]])

# Noise effect matrix in continous-time domain.
L = np.array([[0, 0],
			  [1, 0],
			  [0, 1]])

# Spectral power density of the white noise.
q1 = 0.2
q2 = 0.1
Qc = np.diag([q1, q2])

# Discretize the plant equation.
A, Q = lti_disc(F,L,Qc,dt)

# Generate the real signal.
X = np.zeros(3, n)
X[0] = np.array([f, w, a])[:,None]
for i in range(1,n):
	X[i] = A@X[i-1] + gauss_rnd(np.zeros(3)[:,None], Q)


# Generate the observations with Gaussian noise.
sd = 1
R = sd**2

Y = np.zeros(1,n)
Y_real = h_func(X)     
Y = Y_real + gauss_rnd(0,R,n)


fig = go.Figure()
fig.add_scatter(x=x, y=Y)
fig.add_scatter(x=x, y=Y_real)
fig.show() 


# Initial guesses for the state mean and covariance.
M = np.array([f, w, a])[:,None]
P = np.eye(3) * 3    

# Reserve space for estimates.
m_ = M.shape[0]
y_ = Y.shape[0]

MM = np.zeros(y_,m_,1)
PP = np.zeros(y_,m_,m_)

# Estimate with EKF
for k in range(len(Y)):
	M, P = ekf_predict1(M, P, A, Q)
	M, P = ekf_update1(M, P, Y[k], dh_dx_func, R*np.eye(1), h_func)
	MM[k] = M
	PP[k] = P


# Initial guesses for the state mean and covariance.
M = np.array([f, w, a])[:,None]
P = np.diag([3, 3, 3])    

# Reserve space for estimates.
MM2 = np.zeros(y_,m_,1)
PP2 = np.zeros(y_,m_,m_)

# Estimate with EKF
for k in range(len(Y)):
	M, P = ekf_predict1(M, P, A, Q)
	M, P = ekf_update2(M, P, Y[k], dh_dx_func, d2h_dx2_func, R*np.eye(1), h_func)
	MM2[k]   = M
	PP2[k] = P


'The filtering results using the 1st order EKF is now displayed'

# Project the estimates to measurement space 
Y_m = h_func(MM)
Y_m2 = h_func(MM2)

fig = go.Figure()
fig.add_scatter(x=x, y=Y, name='Measurements')
fig.add_scatter(x=x, y=Y_real, name='Real signal')
fig.add_scatter(x=x, y=Y_m, name='Filtered estimate')
fig.layout.update(title='Estimating a random Sine signal with extended Kalman filter', xaxis_range=[0,max(x)])
fig.show()



'The filtering result using the 1st order EKF is now displayed.'
print('Smoothing the estimates using the RTS smoother...')

SM1,SP1 = erts_smooth1(MM, PP, A, Q)

SM2,SP2 = etf_smooth1(MM, PP, Y, A, Q, None, None, None, dh_dx_func, R*np.eye(1), h_func)  

SM1_2,SP1_2 = erts_smooth1(MM2, PP2, A, Q)

SM2_2,SP2_2 = etf_smooth1(MM2, PP2, Y, A, Q, None, None, None, dh_dx_func, R*np.eye(1), h_func)  

print('ready.\n')

'Push any button to display the smoothing results.'



Y_s1 = h_func(SM1)

fig = go.Figure()
fig.add_scatter(x=x, y=Y, name='Measurements')
fig.add_scatter(x=x, y=Y_real, name='Real signal')
fig.add_scatter(x=x, y=Y_s1, name='Smoothed estimate')
fig.layout.update(title='Smoothing a random Sine signal with Extended Kalman (RTS) smoother', xaxis_range=[0,max(x)])
fig.show()


'The smoothing results using the ERTS smoother is now displayed.'

'Push any button to see the smoothing results of a ETF smoother.'


Y_s2 = h_func(SM2)

fig = go.Figure()
fig.add_scatter(x=x, y=Y, name='Measurements')
fig.add_scatter(x=x, y=Y_real, name='Real signal')
fig.add_scatter(x=x, y=Y_s2, name='Smoothed estimate')
fig.layout.update(title='Smoothing a random Sine signal with Extended Kalman (Two Filter) smoother', xaxis_range=[0,max(x)])
fig.show()


'The smoothing results using the ETF smoother is now displayed.'


Y_s1_2 = h_func(SM1_2)
Y_s2_2 = h_func(SM2_2)
# Errors.  
EMM_Y = np.sum((Y_m-Y_real)**2)/n
EMM_Y2 = np.sum((Y_m2-Y_real)**2)/n
ESM1_Y = np.sum((Y_s1-Y_real)**2)/n
ESM2_Y = np.sum((Y_s2-Y_real)**2)/n
ESM1_2_Y = np.sum((Y_s1_2-Y_real)**2)/n
ESM2_2_Y = np.sum((Y_s2_2-Y_real)**2)/n


print('Filtering now with UKF...')

# In the rest the signal is filtered with UKF for comparison.

# Initial guesses for the state mean and covariance.
M = np.array([f, w, a])[:,None]
P = np.diag([3, 3, 3])    

# Reserve space for estimates.
U_MM = np.zeros(y_,m_,1)
U_PP = np.zeros(y_,m_,m_)

# Estimate with UKF
for k in range(len(Y)):
	M, P, X_s, w = ukf_predict3(M, P, f_func, Q, R*np.eye(1), dt)
	M, P = ukf_update3(M, P, Y[k], h_func, R*np.eye(1), X_s, w, None)
	U_MM[k] = M
	U_PP[k] = P

U_SM, U_SP = urts_smooth1(U_MM, U_PP, f_func, Q, dt)

print('ready.\n')

'Push any button to see the filtering results.'


Y_m_u = h_func(U_MM)

fig = go.Figure()
fig.add_scatter(x=x, y=Y, name='Measurements')
fig.add_scatter(x=x, y=Y_real, name='Real signal')
fig.add_scatter(x=x, y=Y_m_u, name='Filtered estimate')
fig.layout.update(title='Estimating a random Sine signal with unscented Kalman filter', xaxis_range=[0,max(x)])
fig.show()


'The filtering results of a UKF are now displayed.'

'Push any button to display the smoothing results.'


Y_m_su = h_func(U_SM)

fig = go.Figure()
fig.add_scatter(x=x, y=Y, name='Measurements')
fig.add_scatter(x=x, y=Y_real, name='Real signal')
fig.add_scatter(x=x, y=Y_m_su, name='Filtered estimate')
fig.layout.update(title='Estimating a random Sine signal with unscented Kalman smoother (RTS)', xaxis_range=[0,max(x)])
fig.show()

'The smoothing results of a ERTS smoother are now displayed'

UKF_EMM_Y = np.sum((Y_m_u-Y_real)**2)/n
URTS_EMM_Y = np.sum((Y_m_su-Y_real)**2)/n


'Mean square errors of all estimates:'
print('EKF1-MSE = %.4f\n',np.sqrt(EMM_Y))
print('ERTS-MSE = %.4f\n',np.sqrt(ESM1_Y))
print('ETF-MSE = %.4f\n',np.sqrt(ESM2_Y))
print('EKF2-MSE = %.4f\n',np.sqrt(EMM_Y2))
print('ERTS2-MSE = %.4f\n',np.sqrt(ESM1_2_Y))
print('ETF2-MSE = %.4f\n',np.sqrt(ESM2_2_Y))
print('UKF-RMSE = %.4f\n',np.sqrt(UKF_EMM_Y))
print('URTS-RMSE = %.4f\n',np.sqrt(URTS_EMM_Y))

