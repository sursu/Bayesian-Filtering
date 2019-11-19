import numpy as np

def bot_h(x,s):
    """
    Azimuth measurement function for EKF/UKF.

        h = atan((y-sy) / (x-sx))

    Copyright (C) 2003 Simo S�rkk�

    This software is distributed under the GNU General Public 
    Licence (version 2 or later); please refer to the file 
    Licence.txt, included with the software, for details.
    """

    Y = np.zeros((s.shape[1], x.shape[1]))

    for i in range(s.shape[1]):
        h = np.arctan2(x[1,:]-s[1,i], x[0,:]-s[0,i])
        _p = (h >  0.5*np.pi)
        _n = (h < -0.5*np.pi)
        if np.sum(_n) > np.sum(_p):
            h[_p] -= 2*np.pi
        else:
            h[_n] += 2*np.pi
        Y[i] = h

    return Y