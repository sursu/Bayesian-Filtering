# GAUSS_PDF  

# Copyright (C) 2002 Simo Särkkä
#
# $Id$
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

from numpy import tile, sum, log, exp, pi, matrix, multiply
from numpy.linalg import solve, det

def gauss_pdf(X, M, S):
    """
    Multivariate Gaussian PDF

    Syntax:
    [P,E] = GAUSS_PDF(X,M,S)

    In:
    X - Dx1 value or N values as DxN matrix
    M - Dx1 mean of distibution or N values as DxN matrix.
    S - DxD covariance matrix

    Out:
    P - Probability of X. 
    E - Negative logarithm of P
    
    Description:
    Calculate values of PDF (Probability Density
    Function) of multivariate Gaussian distribution

    N(X | M, S)

    Function returns probability of X in PDF. If multiple
    X's or M's are given (as multiple columns), function
    returns probabilities for each of them. X's and M's are
    repeated to match each other, S must be the same for all.

    See also:
        GAUSS_RND
    """

    d = matrix(M).shape[0]
    m_ = matrix(M).shape[1]
    x_ = matrix(X).shape[1]

    if m_ == 1:
        DX = X - tile(M, (1, x_))
        E = 0.5 * sum(multiply(DX,solve(S,DX)), axis=0)
        E += 0.5 * (d * log(2*pi) + log(det(S)))
        P = exp(-E)
    elif x_ == 1:
        DX = tile(X, (1, m_)) - M
        E = 0.5*sum(multiply(DX,solve(S,DX)), axis=0)
        E += 0.5 * (d * log(2*pi) + log(det(S)))
        P = exp(-E)
    else:
        DX = X - M
        E = 0.5*DX.T@solve(S,DX)
        E += 0.5 * (d * log(2*pi) + log(det(S)))
        P = exp(-E)

    return P, E