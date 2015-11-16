import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Plot3DImage(rgb_image,depth_image,sample=True,samplerate=.5,ignoreOutliers=True):
    """
    Input: 
         rgb_image[pixely,pixelx,rgb] - Standard rgb image [0,255]
         depth_image[pixely,pixelx,depth] - Image storing depth measurements as greyscale - 255=5m, 0=0m
         sample(Bool) - Plot full image or random sample of points
         samplerate(0,1) - Percentage of pixels to plot
         ignoreOutliers - Ignores values at the camera lens (0,0,0)
    Output:
         None (Creates a matplotlib plot)
    """

    XYZ=depth2XYZ(depth_image)
    flatXYZ=XYZ.reshape((XYZ.shape[0]*XYZ.shape[1],XYZ.shape[2]))
    flatColors=rgb_image.reshape((rgb_image.shape[0]*rgb_image.shape[1],rgb_image.shape[2]))/255.0

    plotIndices=(np.ones((rgb_image.shape[0]*rgb_image.shape[1]))==1)

    if(ignoreOutliers):
        zeroIndices=(flatXYZ[:,0]**2+flatXYZ[:,1]**2+flatXYZ[:,2]**2)>0
        plotIndices=plotIndices & zeroIndices
    if(sample):
        plotIndices=plotIndices & (np.random.rand(plotIndices.shape[0])<=samplerate)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(flatXYZ[plotIndices==True,0],flatXYZ[plotIndices==True,1],flatXYZ[plotIndices==True,2],c=flatColors[plotIndices==True,:], s=8, edgecolors='none')
    ax.view_init(elev=-115, azim=-90)
    plt.show()
    
    return(None)

def depth2XYZ(depth_image, freiburg1=True):
    """
    Input: 
         depth_image[pixely,pixelx,depth] - Image storing depth measurements as greyscale - 255=5m, 0=0m
    Output:
         XYZ[pixely,pixelx,xyz] - Returns a matrix of three dimensional coordinates
    """

    depth_image=depth_image[:,:,0]

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


    XYZ=np.zeros((depth_image.shape[0],depth_image.shape[1],3))

    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            XYZ[v,u,2]=depth_image[v,u]/factor;
            XYZ[v,u,0]=(u-cx)*XYZ[v,u,2]/fx;
            XYZ[v,u,1]=(v-cy)*XYZ[v,u,2]/fy;

    return(XYZ)


testrgbimg=cv2.imread("3DtestRGB.png")
testdepthimg=cv2.imread("3DtestD.png")

Plot3DImage(testrgbimg,testdepthimg)
