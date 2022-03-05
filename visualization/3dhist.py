#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Generates 3D Histogram of Wallaby image and renders to screen using vispy

Requires:

vispy
scipy
numpy

Related URLs:

http://vispy.org/installation.html
https://github.com/vispy/vispy/blob/master/examples/basics/scene/volume.py
http://api.vispy.org/en/latest/scene.html#vispy.scene.visuals.Volume
'''

from urllib.request import urlopen

import numpy as np
# from scipy.misc.pilutil import imread
import cv2

from vispy import scene
from vispy import app
from vispy.io import load_data_file, read_png

# Create the all zero 3D Histogram we will use to store the color information
tristogram = np.zeros((256,256,256), dtype=np.uint8)


# url = "https://raw.githubusercontent.com/desertpy/presentations/master/exploring-numpy-godber/wallaby_746_600x450.jpg"
path = r"C:\Users\Heet\Desktop\CVPR\Real-Time-Object-2D-Recognition\Data\Examples\img1p3.png"
img = cv2.imread(path)

for h in range(img.shape[0]):
    for w in range(img.shape[1]):
        (b,g,r) = img[h, w, :]
        tristogram[r, g, b] += 1

canvas = scene.SceneCanvas(show=True)
view = canvas.central_widget.add_view()
volume = scene.visuals.Volume(tristogram, parent=view.scene, emulate_texture=False)
view.camera = scene.cameras.TurntableCamera(parent=view.scene)


if __name__ == '__main__':
    app.run()
