from sphericalradial import sphericalradial
from ckf_packed_pc import ckf_packed_pc

def ckf_transform(m, P, g, param=None, varargin=None):
    """
    CKF_TRANSFORM - Cubature Kalman filter transform of random variables
    
    Syntax:
      [mu,S,C,SX,W] = CKF_TRANSFORM(M,P,g,param)
    
    In:
      M - Random variable mean (Nx1 column vector)
      P - Random variable covariance (NxN pos.def. matrix)
      g - Transformation function of the form g(x,param) as
          matrix, inline function, function name or function reference
      g_param - Parameters of g               (optional, default empty)
    
    Out:
      mu - Estimated mean of y
       S - Estimated covariance of y
       C - Estimated cross-covariance of x and y
      SX - Sigma points of x
       W - Weights as cell array

    Copyright (c), 2010 Arno Solin
    
    This software is distributed under the GNU General Public 
    Licence (version 2 or later); please refer to the file 
    Licence.txt, included with the software, for details.
    """

    # Estimate the mean of g
    if param is None:
        mu, SX, W, _ = sphericalradial(g, m, P)
    else:
        mu, SX, W, _ = sphericalradial(g, m, P, param)


    # Estimate the P and C
    if param is None:
        pc, SX, W, _ = sphericalradial(ckf_packed_pc, m, P, [g,m,mu])
    else:
        pc, SX, W, _ = sphericalradial(ckf_packed_pc, m, P, [g,m,mu,param])

    d = m.shape[0]
    s = mu.shape[0]
    S = pc[:s**2].reshape((s,s))
    C = pc[s**2:].reshape((d,s))
    
    return mu, S, C, SX, W