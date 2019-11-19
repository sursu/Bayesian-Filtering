# KF_PREDICT

# History:
#
# 26.2.2007 JH Added the distribution model for the predicted state
#             and equations for calculating the predicted state mean and
#             covariance to the description section.

# Copyright (C) 2002-2006 Simo Särkkä
# Copyright (C) 2007 Jouni Hartikainen
#
# $Id$
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

import numpy as np

def kf_predict(x, P, A=None, Q=None, B=None, u=None):
    """
    Perform Kalman Filter prediction step

    Syntax:
    X, P = kf_predict(X,P,A,Q,B,U)

    In:
    X - Nx1 mean state estimate of previous step
    P - NxN state covariance of previous step
    A - Transition matrix of discrete model (optional, default identity)
    Q - Process noise of discrete model     (optional, default zero)
    B - Input effect matrix                 (optional, default identity)
    U - Constant input                      (optional, default empty)

    Out:
    X - Predicted state mean
    P - Predicted state covariance
    
    Description:
    Perform Kalman Filter prediction step. The model is

        x[k] = A*x[k-1] + B*u[k-1] + q,  q ~ N(0,Q).

    The predicted state is distributed as follows:
    
        p(x[k] | x[k-1]) = N(x[k] | A*x[k-1] + B*u[k-1], Q[k-1])

    The predicted mean x-[k] and covariance P-[k] are calculated
    with the following equations:

        m-[k] = A*x[k-1] + B*u[k-1]
        P-[k] = A*P[k-1]*A' + Q.

    If there is no input u present then the first equation reduces to
        m-[k] = A*x[k-1]
    
    See also:
        KF_UPDATE, LTI_DISC, EKF_PREDICT, EKF_UPDATE
    """

    # Apply defaults
    if A is None:
        A = np.eye(len(x))
    if Q is None:
        Q = np.zeros(len(x))
    if B is None and u is not None:
        B = np.eye((len(x), len(u)))

    # Perform prediction
    if u is None:
        x = A @ x
        P = A @ P @ A.T + Q
    else:
        x = A @ x + B * u
        P = A @ P @ A.T + Q

    return x, P