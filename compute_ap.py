import json
import os

import cPickle
import numpy as np
import sys
from absl import app
from absl import flags

home = os.getenv('HOME')
sys.path.insert(0, home + '/lab/ActivityNet/Evaluation/ava')
import np_box_list
import np_box_list_ops
import metrics

flags.DEFINE_string('subset', 'val', 'subset')

flags.DEFINE_string('job_dir', 'saving', 'job dir')

FLAGS = flags.FLAGS


def get_overlaps_and_scores_box_mode(
    detected_boxes,
    detected_scores,
    groundtruth_boxes):
  """Computes overlaps and scores between detected and groudntruth boxes.

  Args:
    detected_boxes: A numpy array of shape [N, 4] representing detected box
        coordinates
    detected_scores: A 1-d numpy array of length N representing classification
        score
    groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
        box coordinates

  Returns:
    iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
        gt_non_group_of_boxlist.num_boxes() == 0 it will be None.
    scores: The score of the detected boxlist.
    num_boxes: Number of non-maximum suppressed detected boxes.
  """
  detected_boxlist = np_box_list.BoxList(detected_boxes)
  detected_boxlist.add_field('scores', detected_scores)
  detected_boxlist = np_box_list_ops.non_max_suppression(
    detected_boxlist, 10000, 1.0)
  gt_non_group_of_boxlist = np_box_list.BoxList(groundtruth_boxes)
  iou = np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
  scores = detected_boxlist.get_field('scores')
  num_boxes = detected_boxlist.num_boxes()
  return iou, scores, num_boxes


def main(_):
  with open('%s/ret_%s.pkl' % (FLAGS.job_dir, FLAGS.subset), 'r') as f:
    ret = cPickle.load(f)
  with open('data/%s.json' % FLAGS.subset, 'r') as f:
    gt = json.load(f)
  for th in [0.3, 0.5, 0.7]:
    all_scores = {}
    all_tp_fp = {}
    num_gt_instances = {}
    for pred, g in zip(ret, gt):
      ref = g[1]
      label = tuple(g[0]['label'])
      if label not in num_gt_instances:
        num_gt_instances[label] = 0
      n = pred['num_detections'].astype(np.int32)
      for i, gt_boxes in enumerate(ref['gt']):
        if len(gt_boxes) > 0:
          gt_boxes = np.asarray(gt_boxes)
        else:
          gt_boxes = np.zeros((0, 4), np.float32)
        num_gt_instances[label] += gt_boxes.shape[0] * 3
        for j in range(3):
          idx = i * 3 + j
          det_boxes = pred['detection_boxes'][idx][:n[idx]]
          det_scores = pred['detection_scores'][idx][:n[idx]]
          det_classes = pred['detection_classes'][idx][:n[idx]]
          mask = det_classes == 1
          det_boxes = det_boxes[mask]
          det_scores = det_scores[mask]
          iou, scores, num_detected_boxes = get_overlaps_and_scores_box_mode(
            det_boxes,
            det_scores,
            gt_boxes)

          tp_fp_labels = np.zeros(det_scores.size, dtype=bool)
          if iou.size > 0:
            max_overlap_gt_ids = np.argmax(iou, axis=1)
            is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
            for k in range(num_detected_boxes):
              gt_id = max_overlap_gt_ids[k]
              if iou[k, gt_id] >= th:
                if not is_gt_box_detected[gt_id]:
                  tp_fp_labels[k] = True
                  is_gt_box_detected[gt_id] = True
          all_scores.setdefault(label, []).append(scores)
          all_tp_fp.setdefault(label, []).append(tp_fp_labels)

    aps = []
    for k in all_scores:
      scores = np.concatenate(all_scores[k])
      tp_fp_labels = np.concatenate(all_tp_fp[k])
      precision, recall = metrics.compute_precision_recall(
        scores, tp_fp_labels, num_gt_instances[k])
      average_precision = metrics.compute_average_precision(precision, recall)
      aps.append(average_precision)
    mean_ap = np.mean(aps)
    print(th, mean_ap)


if __name__ == '__main__':
  app.run(main)
