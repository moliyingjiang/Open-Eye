from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import torch
import random

from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from models.experimental import attempt_load

from src.lib.opts import opts
from src.lib.detectors.detector_factory import detector_factory

import os
import sys

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np


def init():
    FILE = Path(__file__).absolute()
    sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
    device = torch.device('cuda:0')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load('~/models-open-eye/eye.pt', map_location=device)  # load FP32 model
    imgsz = check_img_size(640, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    img01 = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img01.half() if half else img01) if device.type != 'cpu' else None  # run once
    return device, half, model, names, colors


def predict_img(imgs, device, half, model):
    img = [letterbox(x, new_shape=640, auto=True)[0] for x in imgs]
    # Stack
    img = np.stack(img, 0)
    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[0, 1, 2, 3, 4, 5, 6, 7], agnostic=False)
    return img, pred


# def demo(opt, img):
#     # Run detector on the current frame
#
#     # print result
#     return result
def main(opt):
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), f'Failed to open {0}'
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img, pred = predict_img([frame], device, half, model)
            other_results = []
            yolo_results = []

            for i, det in enumerate(pred):  # detections per image
                if len(det):

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                              frame.shape).round()  # use frame.shape instead of im0.shape
                    for *xyxy, conf, cls in reversed(det):
                        # other_results.append(names[int(cls)])
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        label = f'{names[int(cls)]}'
                        # zzz = True
                        if conf >= 0.6 and names[int(cls)] == 'eye':
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            sign_roi = frame[y1:y2, x1:x2]
                            # alpha = 1.6
                            # beta = 40
                            # new_image = cv2.convertScaleAbs(sign_roi, alpha=alpha, beta=beta)
                            direction = detector.run(sign_roi)
                            # other_results.append(direction)

                            if 'higher' in direction:
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                              (0, 0, 255), 3)
                                cv2.putText(frame, direction, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0, (0, 0, 255), 3)
                                cv2.putText(frame, 'Higher!', (10, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            if 'lower' in direction:
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                              (0, 255, 255), 3)
                                cv2.putText(frame, direction, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0, (0, 255, 255), 3)
                                cv2.putText(frame, 'low!', (10, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
                            if 'normal' in direction:
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                              (0, 255, 0), 3)
                                cv2.putText(frame, direction, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0, (0, 255, 0), 3)
                                cv2.putText(frame, 'Normal!', (10, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)


                        elif conf >= 0.8:
                            yolo_results.append(label)
                            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)],
                                         line_thickness=3)  # draw boxes on frame

                        total_results = other_results + yolo_results
                        known_labels_en = ["forward", "back", "circle", "mask"]
                        known_labels_cn = ["向前", "后退", "原地扭身", "口罩"]

                        # 创建英文标签与中文标签的映射字典
                        label_mapping = dict(zip(known_labels_en, known_labels_cn))
                        labels_new = []
                        # 遍历随机列表
                        for item in total_results:
                            if item in known_labels_en:
                                # 获取对应的中文标签
                                chinese_label = label_mapping[item]
                                labels_new.append(chinese_label)
                                # print(labels_new)
                        if len(total_results) !=0:
                            if '向前' in labels_new:
                                print('识别到手势为 前进')
                            if '后退' in labels_new:
                                print('识别到手势为 后退')
                            if '原地扭身' in labels_new:
                                print('识别到手势为 原地扭身')


                        # ********

            # Display the frame with bounding boxes and labels
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    device, half, model, names, colors = init()
    opt = opts().init()
    opt.task = 'multi_pose'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt.debug = max(opt.debug, 1)
    opt.task = 'multi_pose'
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    detector.pause = False
    main(opt)
    # import src.lib.opts
