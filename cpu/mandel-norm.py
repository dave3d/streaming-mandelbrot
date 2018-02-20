#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
#from matplotlib.pylab import imshow, jet, show, ion
import numpy as np

from numba import jit, int32, float64, njit, prange
import cv2
import math, cmath

from numba import jit
# color map
cmap = [ 66, 30, 15,   25, 7, 26,    9, 1, 47,      4, 4, 73,      0, 7, 100,     12, 44, 138,
         24, 82, 177,  57, 125, 209, 134, 181, 229, 211, 236, 248, 241, 233, 191,
         248, 201, 95, 255, 170, 0,  204, 128, 0,   153, 87, 0,    106, 52, 3, 106, 52, 3 ]


#@jit(int32(float64,float64,int32), nopython=True)
@jit
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return z, i

    return z, 0

@njit(parallel=True)
def create_fractal(min_x, max_x, min_y, max_y, image, cmap, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    l = len(cmap)
    last_index = l-3

    horizon = 2.0 ** 40
    log_horizon = math.log(math.log(horizon))/math.log(2)
    ilog2 = 1.0/math.log(2.0)

    for x in prange(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            z, n = mandel(real, imag, iters)

            val = n + 1 -  math.log(math.log(abs(z)))*ilog2 + log_horizon
            if math.isnan(val):
              #index = last_index
              image[y,x][2] = 0
              image[y,x][1] = 0
              image[y,x][0] = 0
            else:

              val = int(val)

              # lookup color from val
              index = int(3*val)
              if index>=l:
                index=last_index

              image[y,x][2] = cmap[index]
              image[y,x][1] = cmap[index+1]
              image[y,x][0] = cmap[index+2]

    return image

image = np.zeros((2048, 4096, 3), dtype=np.uint8)
create_fractal(-2.0, 1.0, -1.0, 1.0, image, cmap, 20)


import sys
import os
home = os.environ['HOME']
mandel_dir = home+"/streaming-mandelbrot/"
sys.path.append(mandel_dir)
print (mandel_dir)
print (sys.path)
import path

path.init_path()

write_flag = True

n = 500
s = timer()
for i in range(n):
  win = path.get_current_window(True)
  print (win)
  create_fractal(win[0], win[1], win[2], win[3], image, cmap, 20)

  if write_flag:
    frame_img = cv2.imencode( '.png', image, [cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, 1,
                             cv2.IMWRITE_PNG_STRATEGY_FIXED, 1] )[1].tobytes()

    fname = "mandel-frame.%04d.png" % i
    fp = open(fname, 'wb')
    fp.write(frame_img)
    fp.close()
    sys.stdout.write('.')

e = timer()
print(e - s, (e-s)/n)
#imshow(image)
#jet()
#ion()
#show()

#import SimpleITK as sitk
#simg = sitk.GetImageFromArray(image)
#sitk.WriteImage(simg, "mandel-norm.vtk")
#sitk.Show(simg)

jpeg_img = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, 1, cv2.IMWRITE_PNG_STRATEGY_FIXED, 1])[1].tobytes()

fp = open('mandel-norm.png', 'wb')
fp.write(jpeg_img)
fp.close()

