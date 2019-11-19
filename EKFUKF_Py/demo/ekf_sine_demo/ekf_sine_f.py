from numpy import array

def ekf_sine_f(x, param):
    """
	Dynamical model function for the random sine signal demo

	Copyright (C) 2007 Jouni Hartikainen
	
	This software is distributed under the GNU General Public 
	Licence (version 2 or later); please refer to the file 
	Licence.txt, included with the software, for details.
    """

    dt = param
    A = array([[1, dt, 0],
               [0, 1, 0],
               [0, 0, 1]])
               
    x_n = A@x[:3,:]
    if x.shape[0] == 6 or x.shape[0] == 7:
        x_n[:3,:] += x[3:6,:]

    return x_n
	