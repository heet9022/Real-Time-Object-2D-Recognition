import cv2
import sys
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

path = r"C:\Users\Heet\Desktop\CVPR\Real-Time-Object-2D-Recognition\Data\Examples\img1p3.png"

img = cv2.imread(path) # in bgr 
cv2.imshow("Window", img)
cv2.waitKey(0)
hist = cv2.calcHist([img],[0, 1, 2],None,[16, 16, 16],[0, 255, 0, 255, 0, 255]) # in bgr

ax.scatter3D(hist, ydata, zdata, c=zdata, cmap='Greens');
print(hist.shape)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # in lab 
# hist = cv2.calcHist([img],[0, 1, 2],None,[100, 2*128, 2*128],[0, 100, -128, 127, -128, 127])`# in lab 

# correlation = cv2.HISTCMP_CORREL # compare histograms using correlation
# corr = cv2.compareHist(img, img2, correlation)