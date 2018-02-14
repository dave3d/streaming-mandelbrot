
#  Code adapted from https://github.com/harrism/numba_examples/blob/master/mandelbrot_numba.ipynb

import sys

import numpy as np
import matplotlib.image
import cv2
from timeit import default_timer as timer

from numba import cuda
from numba import *


cmap = [ 66, 30, 15,   25, 7, 26,    9, 1, 47,      4, 4, 73,      0, 7, 100,     12, 44, 138, 
         24, 82, 177,  57, 125, 209, 134, 181, 229, 211, 236, 248, 241, 233, 191, 
         248, 201, 95, 255, 170, 0,  204, 128, 0,   153, 87, 0,    106, 52, 3 ]

def mandel(x, y, max_iters):
    """
      Given the real and imaginary parts of a complex number,
      determine if it is a candidate for membership in the Mandelbrot
      set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
      z = z*z + c
      if (z.real*z.real + z.imag*z.imag) >= 4:
        return i

    return max_iters

mandel_gpu = cuda.jit(device=True)(mandel)



@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, rgb_image, cmap, iters):
  height = rgb_image.shape[0]
  width = rgb_image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX, startY = cuda.grid(2)
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;
  l=len(cmap)

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      val = mandel_gpu(real, imag, iters)
      #cmap_lookup_gpu(image[y, x], rgb_image[y,x] )
      index = 3*val
      if index>l:
        index=l-3
      
      rgb_image[y,x][2] = cmap[index]
      rgb_image[y,x][1] = cmap[index+1]
      rgb_image[y,x][0] = cmap[index+2]

cmap_numba = np.array(cmap, dtype=np.uint8)
cmap_gpu = cuda.to_device(cmap_numba)

#print type(cmap_numba)
rgbimg = np.zeros((2048, 4096, 3), dtype = np.uint8)
blockdim = (32, 8)
griddim = (32,16)


start = timer()
rgb_d_image = cuda.to_device(rgbimg)
frame_count=1
current_path_point=0
nframes_per_path=40


# render 1 frame before timing to force the JIT
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, rgb_d_image, cmap_gpu, 20) 

# path points are cx,cy,rx,ry,#iterations in pixel space
path = [ [2048,1024, 2048,1024,  20], 
         [784,866,   200,100,    20],
         [784,866,   200,100,    20],
         [784,866,   20,10,      20],
         [784,866,   20,10,      30],
         [784,866,   20,10,      30],
	 [2048,1024, 2048,1024,  20],
	 [2048,1024, 2048,1024,  20],
       ]

def lerp(a, b, frac):
  c=[]
  frac1 = 1.0-frac
  for x,y in zip(a,b):
    r = x*frac1 + y*frac
    c.append(r)
  return c

def pathpt_to_window(pathpt):
  cx = pathpt[0]
  cy = pathpt[1]
  xrad = pathpt[2]
  yrad = pathpt[3]

  winxrad = 1.5*xrad/2048.0
  winyrad = yrad/1024.0
  wincx = -2.0 + 1.5*cx/2048.0
  wincy = -1.0 + cy/1024.0

  #print winxrad, winyrad
  #print wincx, wincy

  wxmin = wincx-winxrad
  wxmax = wincx+winxrad
  wymin = wincy-winyrad
  wymax = wincy+winyrad
  return [wxmin, wxmax, wymin, wymax]
  

def mandel_frame(no_frame_gen=False):
  global frame_count, current_path_point, nframes_per_path

  fc = frame_count % nframes_per_path
  if fc==0:
    current_path_point = current_path_point+1
    if current_path_point >= len(path)-1:
      current_path_point=0
    print frame_count

  frac = fc/(nframes_per_path-1.0)
  pixwin = lerp(path[current_path_point], path[current_path_point+1], frac)
  win=pathpt_to_window(pixwin)
  #print frame_count, fc, current_path_point, win

  if len(pixwin)<5:
    niter = 20
  else:
    niter = int(pixwin[4])

  if niter>20:
    print niter


  mandel_kernel[griddim, blockdim](win[0], win[1], win[2], win[3], rgb_d_image, cmap_gpu, niter) 
  rgb_d_image.to_host()

  if not no_frame_gen:
    #jpeg_img = cv2.imencode('.jpg', rgbimg, [cv2.IMWRITE_JPEG_QUALITY, 30])[1].tobytes()
    jpeg_img = cv2.imencode('.png', rgbimg, [cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, 1, cv2.IMWRITE_PNG_STRATEGY_FIXED, 1])[1].tobytes()
    #jpeg_img = cv2.imencode('.ppm', rgbimg )[1].tobytes()
  else:
     jpeg_img = None

  frame_count = frame_count+1
  return jpeg_img

if __name__ == "__main__":

  for x in path:
    print x
    print pathpt_to_window(x)

  print
  for i in range(20):
    frac = i/19.0
    print lerp(path[0], path[1], frac)

  nframes = 99
  write_frames = True
  
  loadtime = timer()
  print "Loaded GPU in %f s" % (loadtime-start)

  frame_count=0
  # render multiple frames
  for i in range(nframes):
    jpeg_img = mandel_frame(True)

    if jpeg_img and write_frames:
      fname = "mandel-frame.%03d.png" % i
      fp = open(fname, 'wb')
      fp.write(jpeg_img)
      fp.close()

  downloadtime = timer()
  print "Render and D/L of %d frames in %f s" % (nframes, (downloadtime-loadtime))
  print "%f fps" % (nframes/(downloadtime-loadtime))
  dt = timer() - start

  print "Mandelbrot created on GPU in %f s" % dt


  #fp = open('mandel.jpg', 'wb')
  #fp.write(jpeg_img)
  #fp.close()

