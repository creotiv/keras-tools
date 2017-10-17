import os
import numpy as np
import math
from scipy.spatial.distance import cdist

def shape_context(points, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
    # info here: https://github.com/creotiv/Python-Shape-Context/blob/master/info/ShapeContexts425.pdf
    
    nbins = nbins_theta * nbins_r
    t_points = len(points)

    # gettinc euclidian distance
    r_array = cdist(points,points)
    # normalizing
    r_array_n = r_array / r_array.mean()

    # create log space
    r_bin_edges = np.logspace(np.log10(r_inner),np.log10(r_outer),nbins_r)
    r_array_q = np.zeros((t_points,t_points), dtype=int)
    # summing occurences in different log space intervals
    # logspace = [0.1250, 0.2500, 0.5000, 1.0000, 2.0000]
    # 0    1.3 -> 1 0 -> 2 0 -> 3 0 -> 4 0 -> 5 1  
    # 0.43  0     0 1    0 2    1 3    2 4    3 5
    for m in xrange(nbins_r):
       r_array_q +=  (r_array_n < r_bin_edges[m])

    fz = r_array_q > 0

    # getting angles in radians
    theta_array = cdist(landmarks, landmarks, lambda u, v: math.atan2((v[1] - u[1]),(v[0] - u[0])))
    # 2Pi shifted because we need angels in [0,2Pi]
    theta_array_2 = theta_array + 2*math.pi * (theta_array < 0)
    # Simple Quantization
    theta_array_q = (1 + np.floor(theta_array_2 /(2 * math.pi / nbins_theta))).astype(int)
    # norming by mass(mean) angle v.0.1 ############################################
    # By Andrey Nikishaev
    # theta_array_delta = theta_array - theta_array.mean()
    # theta_array_delta_2 = theta_array_delta + 2*math.pi * (theta_array_delta < 0)
    # theta_array_q = (1 + np.floor(theta_array_delta_2 /(2 * math.pi / nbins_theta))).astype(int)
    ###############################################################################

    descriptor = np.zeros((t_points,nbins))
    for i in xrange(t_points):
        sn = np.zeros((nbins_r, nbins_theta))
        for j in xrange(t_points):
            if (fz[i, j]):
                sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
        descriptor[i] = sn.reshape(nbins)

    return descriptor      
