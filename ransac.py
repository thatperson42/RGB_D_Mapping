# ransac.py

from scipy import optimize
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import read_data

def get_Orb_Keypoints_XYZ(rgb1, depth1, rgb2, depth2, m=50, fastThreshhold=60):
    """
    Input:
      rgb1 - rgb data for first image
      d1 - depth data for first image
      rgb2 - rgb data for second image
      d2 - depth data for second image
      m - max number of non-zero keypoints
    Output:
      XYZ1 - XYZ Coordinates of keypoints
      XYZ2 - XYZ Coordinates of keypoints
    """
    debug=False
    debugVerbose=False

    #Initialize extractor
    orb=cv.ORB_create()
    orb.setFastThreshold(fastThreshhold)

    #Find keypoints
    kp1,des1=orb.detectAndCompute(rgb1,None)
    kp2,des2=orb.detectAndCompute(rgb2,None)

    if debug:
        print("Total Keypoints (Image1): ", len(kp1))
        print("Total Keypoints (Image2): ", len(kp2))

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    #Collect the top (HARDCODED) 25 keypairs
    kppixels1=np.zeros((len(matches),2))
    kppixels2=np.zeros((len(matches),2))

    #Find matched pixel pairs
    for i in range(len(matches)):
        kppixels1[i,:]=kp1[matches[i].queryIdx].pt
        kppixels2[i,:]=kp2[matches[i].trainIdx].pt

    if debug:
        if debugVerbose:
            print(kppixels1)
            print(kppixels2)
        img3 = cv.drawMatches(rgb1,kp1,rgb2,kp2,matches[:10],None,flags=2)
        plt.imshow(img3),plt.show()

    #Convert to XYZ data
    tmppix1=pixel_2_XYZ(depth1,kppixels1)
    tmppix2=pixel_2_XYZ(depth2,kppixels2)

    #Find only non-zero entries
    nonzero=((tmppix1==0).sum(axis=1)<1)*((tmppix2==0).sum(axis=1)<1)
    XYZ1=tmppix1[nonzero]
    XYZ2=tmppix2[nonzero]

    #Take only the best m pairs, if more available
    if sum(nonzero)>m:
        XYZ1=XYZ1[0:m,]
        XYZ2=XYZ2[0:m,]

    return(XYZ1,XYZ2)


def pixel_2_XYZ(depth_image, pixels, freiburg1=True):
    """
    Input: 
         depth_image[pixely,pixelx, rgb] == depth at pixely, pixelx, color specified by rgb -
                (all rgb have the same values)
                Image storing depth measurements as greyscale - 255=5m, 0=0m
    Output:
         XYZ[i, ] == xyz coordinates for pixel i
         XYZ[i, xyz] == coordinate for xyz of pixel i
             xyz is 0, 1, or 2
    """

    pixels=pixels.round()

    factor=5000.0/255.0 #For greyscale images
    #Set focal parameters
    if(freiburg1):
        #freiburg1 parameters
        fx=517.3  #Focal length x
        fy=516.5  #Focal length y
        cx=318.6  #Optical center x
        cy=255.3  #Optical center y
    else:
        #default parameters
        fx=525.0  #Focal length x
        fy=525.0  #Focal length y
        cx=319.5  #Optical center x
        cy=239.5  #Optical center y

    XYZ=np.zeros((pixels.shape[0],3))

    for i in range(pixels.shape[0]):
        XYZ[i,2]=depth_image[pixels[i,1],pixels[i,0]]/factor
        XYZ[i,0]=(pixels[i,1]-cx)*XYZ[i,2]/fx;
        XYZ[i,1]=(pixels[i,0]-cy)*XYZ[i,2]/fy;

    return(XYZ)

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
    debug=False
    meanX = x.mean(axis=0)
    meanY = y.mean(axis=0)
    translation = meanY - meanX
    x_centered = x - meanX
    y_centered = y - meanY
    if debug:
        print("x_centered")
        print(x_centered)
        print("y_centered")
        print(y_centered)
    # Find how much to rescale the x's. Entrywise multiplication.
    x_scale = np.sqrt((x_centered * x_centered).sum())
    y_scale = np.sqrt((y_centered * y_centered).sum())
    scale_factor = y_scale / x_scale
    x_centered_prime = x_centered * scale_factor
    if debug:
        print("scale_factor")
        print(scale_factor)
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
    # transpose so each column is an x vector, then transpose back at the end
    # x_final = rotation_win.dot(x_final.T).T
    rotation_full = rotation_win.dot(rotation_p)
    if debug:
        print("rotation_win")
        print(rotation_win)
        print("rotation_full")
        print(rotation_full)
        print("shifting by")
        print(rotation_full.dot(-meanX * scale_factor) + meanY)
    #def T(w):
    #    """
    #    T takes a numpy array with 3 entries, spits out another
    #    numpy array with 3 entries (this is a vector in R^3)
    #    """
    #    shift_tmp = (w - meanX) * scale_factor
    #    rot_tmp = rotation_full.dot(shift_tmp)
    #    return(rot_tmp + meanY)
    # Ignore scale_factor
    #    T(x) = Ax + b
    A = rotation_full
    b = meanY - rotation_full.dot(meanX)
    #return(T)
    return(A, b)


def find_error(p_s, p_t, A_d,
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

# TODO: add in w_j here
    second_sum = np.array([np.sqrt(np.linalg.norm(T(p_s[i]) - T(p_t[i])))
                           for i in A_d])
    error = second_sum.sum() / len(A_d)
    return(error)

def find_argmin_T(p_s, p_t, A_d,
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
        return(find_error(p_s, p_t, A_d,
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
    d_cutoff is cutoff distance for us to consider something an inlier
    """
    import random
    iter = 0
    n_inliers = [0] * n_iter
    # initialize T_list
    A_init = np.zeros([3,3])
    b_init = np.zeros(3)
    max_inliers = 0
    while iter < n_iter:
        iter += 1
        n_s = len(cloud_s)
        n_t = len(cloud_t)
        assert n_s == n_t, "clouds not of equal size in ransac()"
        # TODO: replace this random choice with 3 corresponding feature descriptors
        points_inds = random.sample(range(n_s), 3)
#        points_s = np.random.choice(n_s, 3, replace=False)
#        points_t = np.random.choice(n_t, 3, replace=False)
        x_vals = np.array([cloud_s[i] for i in points_inds])
        y_vals = np.array([cloud_t[i] for i in points_inds])
        #x_vals = np.array([cloud_s[i] for i in points_s])
        #y_vals = np.array([cloud_t[i] for i in points_t])

        # Using Horn 1987, Closed-form solution of absolute orientation
        # using unit quaternions.
        #T_init = horn_adjust(x_vals, y_vals)
        A_init_tmp, b_init_tmp = horn_adjust(x_vals, y_vals)

        # TODO: find inliers to the transformation T
        pred_t = A_init_tmp.dot(cloud_s.T).T + b_init_tmp
# TODO: should really be looking at the distance in the projected space!!
        inliers = [np.linalg.norm(pred_t[i,] - clout_t[i,]) < d_cutoff for i in range(n_s)]
        n_inliers = sum(inliers)

        # TODO: do we want to refit on the inliers?
        if n_inliers > max_inliers:
            A_init = A_init_tmp
            b_init = b_init_tmp
            max_inliers = n_inliers

    # TODO: are we using n_inlier_cutoff in this way? Check the paper!
    if max_inliers < n_inlier_cutoff:
        raise Exception('insufficient inliers!')
    max_index = n_inliers.index(max(n_inliers)) 
    # Compute the best transformation T_star
    A, b = find_argmin_T(points_s, points_t, A_d,
                         A_init, b_init)
    # TODO: do I return T corresponding to A and b, or do I return A and b?
    #I think we want A and b here (Anne)
    return(A, b)



#TESTING CODE - Suppressed
"""
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
print("Output")
print(T_tmp(xs[0]) - ys[0])
"""

print("Anne's Work Starts Here")

#Load first two images
firstimg=read_data.rgbData[read_data.pairedData[100][0]][0]
firstdepth=read_data.depthData[read_data.pairedData[100][1]][0]
secondimg=read_data.rgbData[read_data.pairedData[101][0]][0]
seconddepth=read_data.depthData[read_data.pairedData[101][1]][0]
rgb1=cv.imread(firstimg,0)
depth1=cv.imread(firstdepth,0)
rgb2=cv.imread(secondimg,0)
depth2=cv.imread(seconddepth,0)

#Find Keypoints
(XYZ1,XYZ2)=get_Orb_Keypoints_XYZ(rgb1,depth1,rgb2,depth2,fastThreshhold=150)
#ransac(XYZ1,XYZ2,10,10,.5)
#print(firstimg,firstdepth,secondimg,seconddepth)
