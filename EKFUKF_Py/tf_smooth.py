# TF_SMOOTH  

# History:
#   
#   02.8.2007 JH Changed the name to tf_smooth
#   26.3.2007 JH Fixed a bug in backward filter with observations having
#                having more than one dimension.
        
# Copyright (C) 2006 Simo S�rkk�
#               2007 Jouni Hartikainen
#
# $Id$
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

import numpy as np
from numpy.linalg import solve, inv

import kf_update, kf_predict

def tf_smooth(M, P, Y, A, Q, H, R, use_inf=True):
    """
    Two filter based Smoother

    Syntax:
    [M,P] = TF_SMOOTH(M,P,Y,A,Q,H,R,[use_inf])

    In:
    M - NxK matrix of K mean estimates from Kalman filter
    P - NxNxK matrix of K state covariances from Kalman Filter
    Y - Sequence of K measurement as DxK matrix
    A - NxN state transition matrix.
    Q - NxN process noise covariance matrix.
    H - DxN Measurement matrix.
    R - DxD Measurement noise covariance.
    use_inf - If information filter should be used (default 1)

    Out:
    M - Smoothed state mean sequence
    P - Smoothed state covariance sequence
    
    Description:
    Two filter linear smoother algorithm. Calculate "smoothed"
    sequence from given Kalman filter output sequence
    by conditioning all steps to all measurements.

    Example:
        m = m0;
        P = P0;
        MM = zeros(size(m,1),size(Y,2));
        PP = zeros(size(m,1),size(m,1),size(Y,2));
        for k=1:size(Y,2)
            [m,P] = kf_predict(m,P,A,Q);
            [m,P] = kf_update(m,P,Y(:,k),H,R);
            MM(:,k) = m;
            PP(:,:,k) = P;
        end
        [SM,SP] = tf_smooth(MM,PP,A,Q,H,R,Y);

    See also:
        KF_PREDICT, KF_UPDATE
    """

    M = M.copy()
    P = P.copy()

    m_ = M.shape
    p_ = P.shape
    
    # Run the backward filter
    if use_inf:
        zz = np.zeros(m_)
        SS = np.zeros(p_)
        IR = inv(R)
        IQ = inv(Q)
        z = np.zeros((m_[1],1))
        S = np.zeros((m_[1],m_[1]))
        for k in range(m_[0]-1,-1,-1):
            G = solve(S+IQ, S).T
            S = A.T @ (np.eye(m_[1]) - G) @ S @ A
            z = A.T @ (np.eye(m_[1]) - G) @ z
            zz[k] = z
            SS[k] = S
            S += H.T@IR@H
            z += H.T@IR@Y[k]
    else:
        BM = np.zeros(m_[1])
        BP = np.zeros(p_)
        IA = inv(A)
        IQ = IA@Q@IA.T
        fm = np.zeros((m_[1],1))
        fP = 1e12*np.eye(m_[1])
        BM[:] = fm
        BP[:] = fP
        for k in range(m_[0]-2,-1,-1):
            fm, fP, *_ = kf_update.kf_update(fm, fP, Y[k+1], H, R)
            fm, fP = kf_predict.kf_predict(fm, fP, IA, IQ)
            BM[k] = fm
            BP[k] = fP

    # Combine estimates
    if use_inf:
        for k in range(m_[0]-1):
            G = P[k] @ solve((np.eye(m_[1]) + P[k] @ SS[k]).T, SS[k].T)
            P[k] = inv(inv(P[k]) + SS[k])
            M[k] = M[k] + P[k] @ zz[k] - G @ M[k]
    else:
        for k in range(m_[0]-1):
            tmp = inv(inv(P[k]) + inv(BP[k]))
            M[k] = tmp @ (solve(P[k], M[k]) + solve(BP[k],BM[k]))
            P[k] = tmp

    return M, P

