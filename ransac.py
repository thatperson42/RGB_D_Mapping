# ransac.py

import numpy as np

def ransac(cloud_s, cloud_t, n_iter, n_inlier_cutoff, d_cutoff):
    """
    cloud_s is an array of points in group s (source)
    cloud_t is an array of points in group t (target)
    Each point is a 3-tuple in x,y,z space.
    n_iter is how many iterations to perform (see slide 58, lecture 11)
    n_inlier_cutoff is how many inliers we need to refit
    d_cutoff is cutoff distance for us to consider somethign an inlier
    """
    iter = 0
    n_inliers = [0] * n_iter
# initialize T_list
    while iter < n_iter:
        iter += 1
        n_s = len(cloud_s)
        points_s = np.random.choice(n_s, 3, replace=False)
        points_t = np.random.choice(n_t, 3, replace=False)
# TODO: calculate initial transformation T
# Using Horn 1987, Closed-form solution of absolute orientation
# using unit quaternions.

# TODO: find inliers to the transformation T

        if TODO > n_inlier_cutoff:
# TODO: recompute LS estimate on inliers

# TODO: update below
        #n_inliers[iter] =

   max_index = n_inliers.index(max(n_inliers)) 

# TODO: return the T corresponding to max_index, along with
# the number of inliers


