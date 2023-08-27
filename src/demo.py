from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import _init_paths
from progress.bar import Bar

import os
import cv2

from logger import Logger
from lib.datasets.dataset_factory import dataset_factory
from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # opt.debug = max(opt.debug, 1)

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)

    dataset = Dataset(opt, 'val')

    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    results = {}

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        num_iters = len(image_names)
        bar = Bar('{}'.format(opt.exp_id), max=num_iters)

        for ind, (image_name) in enumerate(image_names):
            img_id = image_name.split("/")[-1].split(".")[0]
            img_path = os.path.join(dataset.img_dir, str(img_id) + ".jpg")
            # img_path = '../images'
            ret = detector.run(img_path)
            results[img_id] = ret['results']
            Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

            time_str = '[{0}/{1}]|'.format(ind, num_iters)

            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            bar.next()

        bar.finish()
        dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
