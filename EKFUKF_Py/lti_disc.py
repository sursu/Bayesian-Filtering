# LTI_DISC  

# History:
# 11.01.2003  Covariance propagation by matrix fractions
# 20.11.2002  The first official version.

# Copyright (C) 2002, 2003 Simo Särkkä
#
# $Id$
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

import numpy as np
from numpy.linalg import solve
from scipy.linalg import expm

def lti_disc(F, L=None, Q=None, dt=None):
    """
    Discretize LTI ODE with Gaussian Noise

    Syntax:
    [A,Q] = lti_disc(F,L,Qc,dt)

    In:
    F  - NxN Feedback matrix
    L  - NxL Noise effect matrix        (optional, default identity)
    Qc - LxL Diagonal Spectral Density  (optional, default zeros)
    dt - Time Step                      (optional, default 1)

    Out:
    A - Transition matrix
    Q - Discrete Process Covariance

    Description:
    Discretize LTI ODE with Gaussian Noise. The original
    ODE model is in form

        dx/dt = F x + L w,  w ~ N(0,Qc)

    Result of discretization is the model

        x[k] = A x[k-1] + q, q ~ N(0,Q)

    Which can be used for integrating the model
    exactly over time steps, which are multiples
    of dt.
    """    

    n = F.shape[0]
    if L is None:
        L = np.eye(n)
    if Q is None:
        Q = np.zeros(shape=(n,n))
    if dt is None:
        dt = 1


    # Closed form integration of transition matrix
    A = expm(F*dt)

    # Closed form integration of covariance
    # by matrix fraction decomposition
    Phi = np.block([[F, L@Q@L.T], [np.zeros((n,n)), -F.T]])
    AB  = expm(Phi*dt) @ np.vstack([np.zeros((n,n)), np.eye(n)])
    Q   = solve(AB[n:2*n].T, AB[:n].T)
    
    return A, Q
