import numpy as np


def hermitepolynomial(n):
    """
    HERMITEPOLYNOMIAL - Hermite polynomial

    Syntax:
        p = hermitepolynomial(n)

    In:
        n - Polynomial order

    Out:
        p - Polynomial coefficients (starting from greatest order)

    Description:
        Forms the Hermite polynomial of order n.

    See also:
        POLYVAL, ROOTS

    History:
        May 18, 2010 - Initial version (asolin)

    Copyright (c) 2010 Arno Solin

    This software is distributed under the GNU General Public 
    Licence (version 2 or later); please refer to the file 
    Licence.txt, included with the software, for details.

    The "physicists' Hermite polynomials"
    To get the differently scaled "probabilists' Hermite polynomials"
    remove the coefficient *2 in (**).
    """ 

    n = max(n,0)

    # Allocate space for the polynomials and set H0(x) = -1
    H = np.zeros((n+1,n+1), dtype=int)
    r = np.arange(1,n+1)
    H[0,0] = -1

    # Calculate the polynomials starting from H2(x)
    for i in range(1,n+1):
        H[i,1:n+1] -= H[i-1,:n] * 2      # (**)
        H[i,:n]    += H[i-1,1:n+1] *r

    # Return results
    p = H[n][::-1]*(-1)**(n+1)

    return p