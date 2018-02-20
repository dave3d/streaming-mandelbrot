#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
#from matplotlib.pylab import imshow, jet, show, ion
import numpy as np

from numba import jit, int32, float64, njit, prange
import cv2

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
            return i

    return 255

@njit(parallel=True)
def create_fractal(min_x, max_x, min_y, max_y, image, cmap, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    l = len(cmap)
    for x in prange(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            val = mandel(real, imag, iters)

            # lookup color from val
            index = 3*val
            if index>l:
              index=l-3

            image[y,x][2] = cmap[index]
            image[y,x][1] = cmap[index+1]
            image[y,x][0] = cmap[index+2]

    return image

image = np.zeros((2048, 4096, 3), dtype=np.uint8)
create_fractal(-2.0, 1.0, -1.0, 1.0, image, cmap, 20)
n = 100
s = timer()
for i in range(n):
  create_fractal(-2.0, 1.0, -1.0, 1.0, image, cmap, 20)
e = timer()
print(e - s, (e-s)/n)
#imshow(image)
#jet()
#ion()
#show()

jpeg_img = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, 1, cv2.IMWRITE_PNG_STRATEGY_FIXED, 1])[1].tobytes()

fp = open('mandelbrot.png', 'wb')
fp.write(jpeg_img)
fp.close()

