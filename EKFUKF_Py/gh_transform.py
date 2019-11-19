import numpy as np

from ngausshermi import ngausshermi
from gh_packed_pc import gh_packed_pc

def gh_transform(m, P, g, g_param=None, tr_param=None):
    """
    GH_TRANSFORM - Gauss-Hermite transform of random variables
    
    Syntax:
      [mu,S,C,SX,W] = GH_TRANSFORM(M,P,g,p,param)
    
    In:
      M - Random variable mean (Nx1 column vector)
      P - Random variable covariance (NxN pos.def. matrix)
      g - Transformation function of the form g(x,param) as
          matrix, inline function, function name or function reference
      g_param - Parameters of g               (optional, default empty)
      tr_param - Parameters of the integration method in form {p}:
          p - Number of points in Gauss-Hermite integration
    
    
    Out:
      mu - Estimated mean of y
       S - Estimated covariance of y
       C - Estimated cross-covariance of x and y
      SX - Sigma points of x
       W - Weights as cell array

    Copyright (c), 2009, 2010 Hartikainen, Särkkä, Solin
    
    This software is distributed under the GNU General Public
    Licence (version 2 or later); please refer to the file
    Licence.txt, included with the software, for details.
    """

    if tr_param is not None:
        p = tr_param
    else:
        p = 3

    # Estimate the mean of g
    if g_param is None:
        mu, SX, _, W, _ = ngausshermi(g, p, m, P)
    else:
        mu, SX, _, W, _ = ngausshermi(g, p, m, P, g_param)

    
    # Estimate the P and C
    if tr_param is None:
        pc, SX, _, W, _ = ngausshermi(gh_packed_pc, p, m, P, [g, m, mu])
    else:
        pc, SX, _, W, _ = ngausshermi(gh_packed_pc, p, m, P, [g, m, mu, g_param])

    d = m.shape[0]
    s = mu.shape[0]
    S = pc[:s**2].reshape((s,s))
    C = pc[s**2:].reshape((d,s))

    return mu, S, C, SX, W

  