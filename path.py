#! /usr/bin/env python




frame_count=1
current_path_point=-1
nframes_per_path=50


# path points are [cx,cy,rx,ry,#iterations] in pixel space
path = [ [2048,1024, 2048,1024,  100],
	 [2048,1024, 2048,1024,  100],
         [784,866,   200,100,    100],
         [784,866,   200,100,    100],
         [784,866,   20,10,      100],
         [784,866,   20,10,      200],
         [784,866,   20,10,      200],
         [784,866,   2,1,      200],
         [784,866,   2,1,      200],
	 [2048,1024, 2048,1024,  100],
	 [2048,1024, 2048,1024,  100],
       ]


# bilerp
def lerp(a, b, frac):
  c=[]
  frac1 = 1.0-frac
  for x,y in zip(a,b):
    r = x*frac1 + y*frac
    c.append(r)
  return c

# convert from path point (pixel space) to window/Mandelbrot space
def pathpt_to_window(pathpt):
  cx = pathpt[0]
  cy = pathpt[1]
  xrad = pathpt[2]
  yrad = pathpt[3]

  winxrad = 1.5*xrad/2048.0
  winyrad = yrad/1024.0
  wincx = -2.0 + 1.5*cx/2048.0
  wincy = -1.0 + cy/1024.0

  wxmin = wincx-winxrad
  wxmax = wincx+winxrad
  wymin = wincy-winyrad
  wymax = wincy+winyrad
  return [wxmin, wxmax, wymin, wymax]


def init_path():
  frame_count = 1
  current_path_point = -1

def get_current_window(debug=False):
  # compute current location on flight path
  fc = path.frame_count % path.nframes_per_path
  if fc==0:
    current_path_point = current_path_point+1
    if current_path_point >= len(path)-1:
      current_path_point=0
    print frame_count

  frac = fc/(nframes_per_path-1.0)
  pixwin = lerp(path[current_path_point], path[current_path_point+1], frac)
  win=pathpt_to_window(pixwin)

  if len(pixwin)>4:
    win[4] = pixwin[4]

  return win

