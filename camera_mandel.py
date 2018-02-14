import time
from base_camera import BaseCamera
import mandel


class Camera(BaseCamera):
    """An emulated camera rendering Mandelbrot set images every frame"""

    @staticmethod
    def frames():
        count = 0
        starttime = time.time()

        while True:
            time.sleep(0.02)
            yield mandel.mandel_frame()
            count = count+1
            if (count%100) == 0:
              endtime = time.time()
              dt = endtime-starttime
              fps = 100/dt
              print "fps = ", fps
              starttime=endtime
