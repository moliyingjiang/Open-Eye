from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory

import cv2
import numpy as np
import json
import time, os
import sys
import threading

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    # Use webcam for real-time video stream
    cam = cv2.VideoCapture(0)
    detector.pause = False

    while True:
        ret, img = cam.read()

        if not ret:
            break

        # Run detector on the current frame
        result = detector.run(img)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)