import numpy as np
import cv2 as cv
import ransac as ransac
import read_data
import PlotRGBD_3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

totalframes=30
startframe=647

XYZRGB=np.zeros((1,6))
Atransforms=np.zeros((totalframes,3,3))
Atransforms[0,:,:]=np.eye(3)
btransforms=np.zeros((totalframes,3))

for i in range(startframe,startframe+totalframes-1):
    firstimg=read_data.rgbData[read_data.pairedData[i][0]][0]
    firstdepth=read_data.depthData[read_data.pairedData[i][1]][0]
    secondimg=read_data.rgbData[read_data.pairedData[i+1][0]][0]
    seconddepth=read_data.depthData[read_data.pairedData[i+1][1]][0]
    rgb1=cv.imread(firstimg,0)
    depth1=cv.imread(firstdepth,0)
    rgb2=cv.imread(secondimg,0)
    depth2=cv.imread(seconddepth,0)
    
    (XYZ1,XYZ2) = ransac.get_Orb_Keypoints_XYZ(rgb1,depth1,rgb2,depth2,fastThreshhold=80)
    depth1XYZ, depth2XYZ = ransac.convert_depth(depth1, depth2)

    Atransforms[i-startframe+1,:,:],btransforms[i-startframe+1,:] = ransac.ransac(XYZ1, XYZ2, depth1XYZ, depth2XYZ, Atransforms[i-startframe,:,:],btransforms[i-startframe,:], 50, 5, .1)

Acumulative=np.eye(3)
bcumulative=np.zeros((1,3))
framepixels=np.zeros((totalframes))

for i in range(totalframes):
    img2=read_data.rgbData[read_data.pairedData[startframe+i][0]][0]
    depth2=read_data.depthData[read_data.pairedData[startframe+i][1]][0]
    img=cv.imread(img2)
    b,g,r=cv.split(img)
    img=cv.merge([r,g,b])

    depth=cv.imread(depth2,0)
    
    XYZ=PlotRGBD_3D.depth2XYZ(depth,True,False)    
    inds=(XYZ[:,:,0]==XYZ[:,:,1])
    inds=inds*(XYZ[:,:,1]==XYZ[:,:,2])
    poscol=np.dstack((XYZ,img))
    poscol=poscol[inds==0]
    
    poscol[:,(0,1,2)]=np.dot(Acumulative,poscol[:,(0,1,2)].T).T+bcumulative
    if(i<(totalframes-1)):
        Acumulative=np.dot(Atransforms[i+1,:,:],Acumulative)
        bcumulative=bcumulative+btransforms[i+1,:]
    
    XYZRGB=np.row_stack((XYZRGB,poscol))
    framepixels[i]=poscol.shape[0]

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
inds=range(0,XYZRGB.shape[0],25)
ax.scatter(XYZRGB[inds,0],XYZRGB[inds,1],XYZRGB[inds,2],c=XYZRGB[inds,3:6]/255,s=8,edgecolors='none')
ax.view_init(elev=-115,azim=-90)
plt.show()
