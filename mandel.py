
#  Code adapted from https://github.com/harrism/numba_examples/blob/master/mandelbrot_numba.ipynb

import sys, getopt, math

import numpy as np
import cv2
from timeit import default_timer as timer

from numba import cuda
from numba import *


# color map
cmap = [ 66, 30, 15,   25, 7, 26,    9, 1, 47,      4, 4, 73,      0, 7, 100,     12, 44, 138,
         24, 82, 177,  57, 125, 209, 134, 181, 229, 211, 236, 248, 241, 233, 191,
         248, 201, 95, 255, 170, 0,  204, 128, 0,   153, 87, 0,    106, 52, 3, 106, 52, 3 ]

# core mandelbot set computation
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
        return i, z

    return max_iters, z

mandel_gpu = cuda.jit(device=True)(mandel)


# CUDA Mandelbrot kernel
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
  last_index=l-3

  horizon = 2.0 ** 40
  log_horizon = math.log(math.log(horizon))/math.log(2)
  ilog2 = 1.0/math.log(2.0)


  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y
      n, z = mandel_gpu(real, imag, iters)



        index = last_index
      else:

        val = int(val)

        # lookup color from val
        index = int(3*val)
        if index>l:
          index=last_index

      rgb_image[y,x][2] = cmap[index]
      rgb_image[y,x][1] = cmap[index+1]
      rgb_image[y,x][0] = cmap[index+2]


# setup code
cmap_numba = np.array(cmap, dtype=np.uint8)
cmap_gpu = cuda.to_device(cmap_numba)

rgbimg = np.zeros((2048, 4096, 3), dtype = np.uint8)
blockdim = (32, 8)
griddim = (32,16)


start = timer()
rgb_d_image = cuda.to_device(rgbimg)
debug =  False


# render 1 frame before timing to force the JIT
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, rgb_d_image, cmap_gpu, 20)

import path
path.init_path()

# generate a frame of the mandelbrot rendering
def mandel_frame(gen_image=True, to_cpu=True):

  win=path.get_current_window(debug)

  if len(win)<5:
    niter = 20
  else:
    niter = int(win[4])

  # render mandelbrot image
  mandel_kernel[griddim, blockdim](win[0], win[1], win[2], win[3], rgb_d_image, cmap_gpu, niter)
  if to_cpu:
    rgb_d_image.to_host()
    sys.stdout.write('.')

  # convert to image file format (PNG is the fastest)
  if to_cpu and gen_image:
    #jpeg_img = cv2.imencode('.jpg', rgbimg, [cv2.IMWRITE_JPEG_QUALITY, 30])[1].tobytes()
    jpeg_img = cv2.imencode('.png', rgbimg, [cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, 1, cv2.IMWRITE_PNG_STRATEGY_FIXED, 1])[1].tobytes()
    #jpeg_img = cv2.imencode('.ppm', rgbimg )[1].tobytes()
  else:
     jpeg_img = None

  frame_count = frame_count+1
  return jpeg_img


# Command line version generates frames and write to disk
#
if __name__ == "__main__":

  nframes = 500
  write_frames = False
  gpu_only = False

  try:
    opts, args= getopt.getopt( sys.argv[1:], "hdn:p:wg",
                              [ "help", "debug", "write", "gpu_only"  ] )

  except getopt.GetoptErr, err:
    print (str(err))
    usage()
    sys.exit(1)

  for o, a in opts:
    if o in ("-n"):
        nframes = int(a)
    elif o in ("-p"):
        nframes_per_path = int(a)
    elif o in ("-d", "debug"):
        debug = True
    elif o in ("-w", "write"):
        write_frames = True
    elif o in ("-g", "gpu_only"):
        gpu_only = True
    else:
        print "mandel.py: [options]"
        print ""
        print "  -h, --help      This help message"
        print "  -d, --debug     Print debugging info"
        print "  -n int          Number of frames (default=500)"
        print "  -w, --write     Write out frames (default=False)"
        print "  -g, --gpu_only  GPU timing only (write-out disabled) (default=False)"
        print "  -p int          #frames per path point (default=50)"
        assert False, "unhandled options"



  print ""
  print "Animation path"
  for x in path:
    print x
    print pathpt_to_window(x)



  loadtime = timer()
  print "Loaded GPU in %f s" % (loadtime-start)

  frame_count=0
  # render multiple frames
  for i in range(nframes):
    jpeg_img = mandel_frame(write_frames, not gpu_only)

    if jpeg_img and write_frames:
      fname = "mandel-frame.%04d.png" % i
      fp = open(fname, 'wb')
      fp.write(jpeg_img)
      fp.close()

  rendertime = timer()-loadtime
  print "Render and D/L of %d frames in %f s" % (nframes, rendertime)
  print nframes/rendertime, "fps"
  print rendertime/nframes, "seconds per frame"


