# Cenerate data of discrete-time re-entry mechanics

# Copyright (C) 2005-2006 Simo S�rkk�
#               2007      Jouni Hartikainen
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

import numpy as np

import plotly.graph_objects as go

from reentry_f import reentry_f
from reentry_h import reentry_h

from reentry_param import *


nsteps = round(200/dt)
x = true_m0 + np.sqrt(true_P0) @ np.random.normal(size=(5,1))
X = np.zeros((nsteps,x.shape[0],1))
Y = np.zeros((nsteps,2,1))
T = np.arange(dt,nsteps+dt,dt)

for k in range(nsteps):
    ddt = dt / sim_iter
    for i in range(sim_iter):
        x = reentry_f(x, [ddt,b0,H0,Gm0,R0])
        x += L @ np.sqrt(ddt * true_Qc) @ np.random.normal(size=(3,1))
    
    y = reentry_h(x,[xr,yr]) + np.diag([np.sqrt(vr), np.sqrt(va)]) @ np.random.normal(size=(2,1))
    X[k], Y[k] = x, y


aa = 0.02*np.arange(-1,4.1,0.1)
cx = R0 * np.cos(aa)
cy = R0 * np.sin(aa)


# fig = go.Figure()
# fig.add_scatter(x=X[:,0,0], y=X[:,1,0], name='True')
# fig.add_scatter(x=cx, y=cy, line_color='orange', name='Earth')
# fig.add_scatter(x=[xr], y=[yr], mode='markers', marker_size=12, name='Radar')
# fig.layout.update(height=600, showlegend=False)
# fig.show()
