#!/usr/bin/python

import associate

import numpy as np
import cv2
from matplotlib import pyplot as plt

rgbPicsLoc = "rgb/"
depthPicsLoc = "depth/"

accelData = associate.read_file_list('accelerometer.txt')
depthData = associate.read_file_list('depth.txt')
rgbData = associate.read_file_list('rgb.txt')

pairedData = associate.associate(rgbData, depthData, 0.0, 0.02)
truthData = associate.read_file_list('groundtruth.txt')
rgbData = associate.read_file_list('rgb.txt')

fast = cv2.FastFeatureDetector_create()

img = cv2.imread(rgbData.items()[0][1][0], 0)
kp = fast.detect(img, None)
img2 = img
img2 = cv2.drawKeypoints(img, kp, img2, color=(255,0,0))
#cv2.imshow('image', img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite('drawn_img.png', img2)

def proj(x, F, C, B):
    """
see p.8 of Henry et al
F is focal length of the camera
(C[0], C[1]) is the center of hte image in pixels
C_u == C[0], C_v == C[1]
x == x[0], y == x[1], z == x[2]
B is the baseline b/w in the IR emitter and IR camera sensor
    """
    u = F / x[2] * x[0] + C[0]
    v = F / x[2] * x[1] + C[1]
    d = F / x[2] * B
    return (u, v, d)


def performRANSAC(F_s, F_t):
    """
F_s is features for the source frame,
F_t is features for the target frame.
Output T_star, A_f.
    """
# First select A_f

# Now select T_star by minimizing (5) from Henry et al, p.10

