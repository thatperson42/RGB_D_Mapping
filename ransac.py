# ransac.py

from scipy import optimize
import numpy as np

import read_data

def near_orthog(m):
    """
    near_orthog finds a nearby orthogonal matrix to the matrix m
    """
    w = np.linalg.svd(m)
    return(w[0].dot(w[2]))

def horn_adjust(x, y):
    """
    x and y are numpy arrays consisting of three points in 3d space.
    Each row of x is a point, and each row of y is a point.

    We return a function T that, through rotation, scaling, and 
    translation, comes closest to y by least squares.
    Following Horn 1987.
    """
    meanX = x.mean(axis=0)
    meanY = y.mean(axis=0)
    translation = meanY - meanX
    x_centered = x - meanX
    y_centered = y - meanY
    print("x_centered")
    print(x_centered)
    print("y_centered")
    print(y_centered)
# Find how much to rescale the x's. Entrywise multiplication.
    x_scale = np.sqrt((x_centered * x_centered).sum())
    y_scale = np.sqrt((y_centered * y_centered).sum())
    scale_factor = y_scale / x_scale
    print("scale_factor")
    print(scale_factor)
    x_centered_prime = x_centered * scale_factor
    print("x_centered_prime")
    print(x_centered_prime)
# Find angle to rotate the planes
    x_perp = np.cross(x_centered_prime[0], x_centered_prime[1])
    y_perp = np.cross(y_centered[0], y_centered[1])
# Find rotation matrix to rotate the x plane into the y plane
# Using https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
# https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
# TODO: have to check this section! it's tricky
    x_perp_unit = x_perp / np.linalg.norm(x_perp)
    y_perp_unit = y_perp / np.linalg.norm(y_perp)
    v = np.cross(x_perp_unit, y_perp_unit)
    s = np.linalg.norm(v) # sine of angle between the planes
    c = x_perp_unit.dot(y_perp_unit) # cosine of angle between the planes
    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
# rotation_p works on a column vector
# rotation_p acts on the plane
    rotation_p = np.eye(3) + v_x + v_x.dot(v_x) * (1 - c) / s**2.0
    print("rotation_p")
    print(rotation_p)
# TODO: rotate x_centered_prime properly
# Transpose to make each x a column vector, then transpose back for next part
    x_plane = rotation_p.dot(x_centered_prime.T).T
# Now rotate within the plane, as in Sec. 5 of Horn
    v_y = np.array([[0, -y_perp_unit[2], y_perp_unit[1]],
                    [y_perp_unit[2], 0, -y_perp_unit[0]],
                    [-y_perp_unit[1], y_perp_unit[0], 0]])
    s_win_tmp = np.sum([np.cross(x_plane[i], y_centered[i]) for i in range(3)],
                       axis=0).dot(y_perp_unit)
    c_win_tmp = np.sum([x_plane[i].dot(y_centered[i]) for i in range(3)],
                       axis=0)
    sin_theta = s_win_tmp / np.sqrt(np.linalg.norm(s_win_tmp)**2 +
                                    np.linalg.norm(c_win_tmp)**2)
    cos_theta = c_win_tmp / np.sqrt(np.linalg.norm(s_win_tmp)**2 +
                                    np.linalg.norm(c_win_tmp)**2)
    rotation_win = np.eye(3) + sin_theta * v_y + (1 - cos_theta) * v_y.dot(v_y)
    print("rotation_win")
    print(rotation_win)
# transpose so each column is an x vector, then transpose back at the end
    # x_final = rotation_win.dot(x_final.T).T
    rotation_full = rotation_win.dot(rotation_p)
    print("rotation_full")
    print(rotation_full)
    print("shifting by")
    print(rotation_full.dot(-meanX * scale_factor) + meanY)
    def T(w):
        """
        T takes a numpy array with 3 entries, spits out another
        numpy array with 3 entries (this is a vector in R^3)
        """
        shift_tmp = (w - meanX) * scale_factor
        rot_tmp = rotation_full.dot(shift_tmp)
        return(rot_tmp + meanY)
    return(T)

def find_error(f_s, f_t, A_f, p_s, p_t, A_d, beta,
               A, b):
    """
    find_error calculates the error in eq. 5 of the Henry et al paper.

    A,b determine T: T(x) = A * x + b

    TODO: find n_t^j
    TODO: find w_j
    Uses read_data.proj
    """
    def T(x):
        # TODO: make sure it's A.dot(x), and not A.dot(x.T) !!
        return(A.dot(x) + b)

    first_sum = np.array([np.sqrt(np.linalg.norm(read_data.proj(T(f_s[i])) 
                                  - read_data.proj(T(f_t[i])))) for i in A_f])
# TODO: add in w_j here
    second_sum = np.array([np.sqrt(np.linalg.norm(T(p_s[i]) - T(p_t[i])))
                           for i in A_d])
    error = first_sum.sum() / len(A_f) + second_sum.sum() * beta / len(A_d)
    return(error)

#def find_argmin_T(f_s, f_t, A_f, p_s, p_t, A_d, beta,
def find_argmin_T(f_s, f_t, A_f, p_s, p_t, A_d, beta,
             A, b):
    """
    find_argmin_T does the update in eq. 5 of the Henry et al paper.

    Calculates a numerical derivative

    TODO: find n_t^j
    TODO: find w_j
    Uses find_error() to get the error.
    """
    def f_error(x):
        A_tmp = np.reshape(x[0:9], newshape=(3,3))
        b_tmp = x[9:12]
        return(find_error(f_s, f_t, A_f, p_s, p_t, A_d, beta,
                          A_tmp, b_tmp))
   def flatten(A, b):
        # Flatten out A and b into x_0
        return(np.concatenate((np.reshape(A, newshape=(9,)), b)))
    x_0 = flatten(A, b)
    sol = optimize.root(f_error, x_0, method='lm')
    def expand(x):
        # Un-flattens x into the tuple of A and b
        return(np.reshape(x[0:9], newshape=(3,3)), x[9:12])

    A_tmp, b = expand(sol.x)
# TODO: could do a closer near_orthog rotation if there's much change here
    A = near_orthog(A_tmp)
    return(A, b)
    

def ransac(cloud_s, cloud_t, n_iter, n_inlier_cutoff, d_cutoff):
    """
    cloud_s is a Numpy array of points in group s (source)
    cloud_t is a Numpy array of points in group t (target)
    Each point is a Numpy array in x,y,z space.
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
# TODO: replace this random choice with 3 corresponding feature descriptors
        points_s = np.random.choice(n_s, 3, replace=False)
        points_t = np.random.choice(n_t, 3, replace=False)
        x_vals = np.array([cloud_s[i] for i in points_s])
        y_vals = np.array([cloud_t[i] for i in points_t])
# Using Horn 1987, Closed-form solution of absolute orientation
# using unit quaternions.
# TODO: calculate initial transformation T
# Note: T_init here is a function
        T_init = horn_adjust(x_vals, y_vals)

# TODO: find inliers to the transformation T

        if TODO > n_inlier_cutoff:
# TODO: recompute LS estimate on inliers
#optimize.root, with method='lm'

# TODO: update below
        #n_inliers[iter] =
    max_index = n_inliers.index(max(n_inliers)) 
    # Compute the best transformation T_star
    A, b = find_argmin_T(f_s, f_t, A_f, p_s, p_t, A_d, beta,
                         A_init, b_init)
# TODO: do I return T corresponding to A and b, or do I return A and b?
    return(A, b)


#y = np.array([3.45, 2.5, 6.7])
ys = np.random.rand(3,3)
scale_tmp = np.random.rand()
shift_tmp = np.random.rand(3)
mat_tmp = np.random.rand(3,3)
q, _ = np.linalg.qr(mat_tmp)
#q = np.eye(3)
# Have to transpose for multiplication by q then transpose back
xs = scale_tmp * q.dot(ys.T).T + shift_tmp

T_tmp = horn_adjust(xs, ys)
T_tmp(xs[0]) - ys[0]

