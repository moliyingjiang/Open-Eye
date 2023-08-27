from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch.utils.data as data


class COCOHP(data.Dataset):
    num_classes = 1
    num_joints = 8
    default_resolution = [512, 512]
    mean = np.array([0.566579, 0.337587, 0.278284],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.146827, 0.122277, 0.117914],
                   dtype=np.float32).reshape(1, 1, 3)
    # flip_idx = [[0, 1], [3, 18], [4, 17], [5, 16], [6, 15], [7, 14], [8, 13], [9, 12], [10, 11]]
    flip_idx = [[0, 1], [2, 7], [3, 6],[4,5]]
    def __init__(self, opt, split):
        super(COCOHP, self).__init__()
        self.edges = [[0, 1], [2, 3],[4, 5], [6,7]]

        self.acc_idxs = [1, 2, 3, 4, 5,6,7]
        self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, 'images/val')
        if split == 'val':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'val.json')
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                '{}.json').format(split)
        self.max_objs = 32
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing x-ray {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        csv_results = []
        tag = -1
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    # if score > 0.2:
                    bbox_out = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:21], dtype=np.float32).reshape(-1, 2),
                        np.ones((6, 1), dtype=np.float32)], axis=1).reshape(18).tolist()
                    keypoints = list(map(self._to_float, keypoints))
                    keypoints = np.array(keypoints).reshape([6, 3]).astype(np.float32)
                    keypoints = np.delete(keypoints, -1, axis=1).reshape([1, 12]).tolist()
                    csv_result = [int(image_id)] + keypoints[0]
                    if tag != csv_result[0]:
                        csv_results.append(csv_result)

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score)),
                            "keypoints": keypoints
                        }
                        detections.append(detection)
                    tag = csv_result[0]

        return detections, csv_results

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        results_, csv_results = self.convert_eval_format(results)
        json.dump(results_,
                  open('{}/results.json'.format(save_dir), 'w'), indent=4)
        print("*****************************************************")
        print(np.array(csv_results))
        print("*****************************************************")
        np.savetxt('{}/results.csv'.format(save_dir), np.array(csv_results), delimiter=",", fmt=['%d'] + ['%.2f'] * 18)

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        # coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        # coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
        # coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
