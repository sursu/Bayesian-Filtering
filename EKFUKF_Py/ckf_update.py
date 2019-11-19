from numpy.linalg import solve

from ckf_transform import ckf_transform
from gauss_pdf import gauss_pdf

def ckf_update(M, P, Y, h, R, h_param=None):
    """
    CKF_UPDATE - Cubature Kalman filter update step

    Syntax:
        [M,P,K,MU,S,LH] = CKF_UPDATE(M,P,Y,h,R,param)

    In:
        M  - Mean state estimate after prediction step
        P  - State covariance after prediction step
        Y  - Measurement vector.
        h  - Measurement model function as a matrix H defining
            linear function h(x) = H*x, inline function,
            function handle or name of function in
            form h(x,param)
        R  - Measurement covariance.
        h_param - Parameters of h.

    Out:
        M  - Updated state mean
        P  - Updated state covariance
        K  - Computed Kalman gain
        MU - Predictive mean of Y
        S  - Predictive covariance Y
        LH - Predictive probability (likelihood) of measurement.

    Description:
        Perform additive form spherical-radial cubature Kalman filter (CKF)
        measurement update step. Assumes additive measurement noise.

        Function h should be such that it can be given
        DxN matrix of N sigma Dx1 points and it returns 
        the corresponding measurements for each sigma
        point. This function should also make sure that
        the returned sigma points are compatible such that
        there are no 2pi jumps in angles etc.

    Example:
        h = inline('atan2(x(2,:)-s(2),x(1,:)-s(1))','x','s');
        [M2,P2] = ckf_update(M1,P1,Y,h,R,S);

    See also:
        CKF_PREDICT, CRTS_SMOOTH, CKF_TRANSFORM, SPHERICALRADIAL

    References:
        Arasaratnam and Haykin (2009). Cubature Kalman Filters.
        IEEE Transactions on Automatic Control, vol. 54, no. 5, pp.1254-1269

    Copyright (c) 2010 Arno Solin

    This software is distributed under the GNU General Public 
    Licence (version 2 or later); please refer to the file 
    Licence.txt, included with the software, for details.
    """
    
    # Do transform and make the update
    if h_param is None:
        MU, S, C, _, _ = ckf_transform(M, P, h)
    else:  
        MU, S, C, _, _ = ckf_transform(M, P, h, h_param)

    S += R
    K = solve(S.T, C.T).T
    M += K @ (Y - MU)
    P -= K @ S @ K.T

    if h_param is not None:
        LH = gauss_pdf(Y, MU, S)
        return M, P, K, MU, S, LH

    return M, P, K, MU, S