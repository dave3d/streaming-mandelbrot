import time
from base_camera import BaseCamera


class Camera(BaseCamera):
    """An emulated camera implementation that streams a repeated sequence of
    files 1.jpg, 2.jpg and 3.jpg at a rate of one frame per second."""
    imgs = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]

    @staticmethod
    def frames():
        count = 0
        starttime = time.time()
        while True:

            time.sleep(0.02)
            yield Camera.imgs[count % 3]
            count = count+1
            if (count%100) == 0:
              endtime = time.time()
              dt = endtime-starttime
              fps = 100/dt
              print "fps = ", fps
              starttime=endtime
